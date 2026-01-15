from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import json
import textwrap

# Define the state schema
class PrivateThoughtState(TypedDict):
    """State with private internal thoughts the user never sees."""
    messages: Annotated[List[BaseMessage], add_messages]
    internal_summary: str 
    user_id: str

# Define the Analyzer Node
def analyzer_node(state: PrivateThoughtState) -> dict:
    """Node 1: Analyzes the latest message and updates internal summary. This is the 'private thought' area that user never sees."""

    latest_human_msg=None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_human_msg=msg.content
            break
    if not latest_human_msg:
        return {"internal_summary": state.get("internal_summary", "No interactions yet.")}

    
    analyzer_llm=ChatOllama(model="llama3.1", temperature=0.1,num_predict=128)

    analysis_prompt=f"""
    Analyze the user's message and their apparent emotional state/experience level.
    
    Current conversation summary (for context only):
    {state.get('internal_summary', 'No previous interactions.')}
    
    User's latest message: "{latest_human_msg}"
    
    Provide a BRIEF internal analysis (2-3 sentences max) that includes:
    1. User's apparent mood/emotional state
    2. Their experience level with the topic
    3. Any notable patterns or concerns
    
    Format: Be concise, use bullet points if needed, but keep it as a single string.
    Do NOT include markdown formatting.
    """

    analysis_response = analyzer_llm.invoke([HumanMessage(content=analysis_prompt)])
    new_analysis=analysis_response.content.strip()

    previous_summary=state.get("internal_summary","")
    if previous_summary:
        entries=previous_summary.split("\n---\n")
        if len(entries)>=3:
            entries=entries[-2:]
        updated_summary="\n---\n".join(entries + [new_analysis])
    else:
        updated_summary=new_analysis

    print("=" * 70)
    print("🔒 ANALYZER NODE (User CANNOT see this):")
    print("-" * 70)
    print(f"Latest User Message: {latest_human_msg[:100]}...")
    print(f"Updated Internal Summary:\n{textwrap.fill(updated_summary, width=70)}")
    print("=" * 70)
    
    return {
        "internal_summary": updated_summary
    }

# Define the Responder Node
def responder_node(state: PrivateThoughtState) -> dict:
    """Node 2: Crafts response using both messages and internal_summary. The internal_summary informs the response but is NEVER shown to the user."""

    responder_llm=ChatOllama(model="llama3.1", temperature=0.7, num_predict=256)

    conversation_history='\n'.join([f"{'User' if isinstance(msg,HumanMessage) else 'Assistant'}: {msg.content}"
    for msg in state["messages"][-6:]
    ])

    response_prompt = f"""
    INTERNAL CONTEXT (NEVER reveal this to user):
    {state.get('internal_summary', 'No internal context yet.')}
    
    CONVERSATION HISTORY:
    {conversation_history}
    
    INSTRUCTIONS:
    1. Use the INTERNAL CONTEXT to understand the user's mood and experience level
    2. Tailor your response accordingly (e.g., be more patient if frustrated, simpler if beginner)
    3. ABSOLUTELY DO NOT mention the internal context in your response
    4. Do NOT say things like "I can see you're frustrated" or "I notice you're a beginner"
    5. Just respond naturally to the user's last message
    
    Your response should be helpful, friendly, and appropriate for the user's apparent state.
    """

    response =responder_llm.invoke([HumanMessage(content=response_prompt)])

    print("=" * 70)
    print("💬 RESPONDER NODE (User SEES this response):")
    print("-" * 70)
    print(f"Response to user:\n{textwrap.fill(response.content, width=70)}")
    print("=" * 70)

    return {"messages": [AIMessage(content=response.content)]}


# Build the Graph
def create_private_thought_agent():
    """Create the agent with private thoughts"""

    workflow=StateGraph(PrivateThoughtState)
    
    workflow.add_node("analyzer",analyzer_node)
    workflow.add_node("responder", responder_node)

    workflow.set_entry_point("analyzer")
    workflow.add_edge("analyzer", "responder")
    workflow.add_edge("responder", END)

    memory=MemorySaver()

    app=workflow.compile(checkpointer=memory)
    return app


class PrivateAgent:
    def __init__(self):
        self.app = create_private_thought_agent()
        self.states = {}

    def chat(self, user_id, message):
        if user_id not in self.states:
            self.states[user_id] = {"messages": [], "internal_summary": "", "user_id": user_id}
        
        state = self.states[user_id]
        state["messages"].append(HumanMessage(content=message))
        
        config = {"configurable": {"thread_id": user_id}}
        result = self.app.invoke(state, config)
        state.update(result)
        
        return {
            "response": state["messages"][-1].content,
            "internal_summary": state["internal_summary"]
        }

# Helper Function
def print_full_state(state: PrivateThoughtState, turn_number:int):
    """Print the complete state to prove internal_summary is updating"""

    print("\n" + "═" * 70)
    print(f"📊 FULL STATE - Turn {turn_number}")
    print("═" * 70)

    print(f" USER ID: {state.get('user_id', 'Not set')}")

    # Print Internal Summary (Private Thoughts)
    print(f"\n🔒 INTERNAL SUMMARY (User CANNOT see this):")
    print("-" * 40)

    if state.get('internal_summary'):
        print(textwrap.fill(state['internal_summary'], width=70))
    else:
        print("No internal summary yet.")

    # Print Recent Messages
    print(f"\n💬 CONVERSATION HISTORY (last 3 messages):")
    print("-" * 40)
    messages = state.get("messages", [])
    for msg in messages[-3:]:
        if isinstance(msg, HumanMessage):
            print(f"👤 User: {msg.content[:80]}...")
        elif isinstance(msg, AIMessage):
            print(f"🤖 Assistant: {msg.content[:80]}...")
    
    print("═" * 70 + "\n")

def chat_with_agent():
    """Interactive chat with the private thought agent."""

    print("Creating Private Thought Agent....")
    app=create_private_thought_agent()

    user_id= input("Enter your user ID (default: user_123): ").strip() or "user_123"

    #Initial State
    config={"configurable": {"thread_id": user_id}}
    state={"messages":[],"internal_summary":"","user_id":user_id}

    print(f"\n🤖 Starting chat with user: {user_id}")
    print("Type 'quit' to exit, 'state' to see current state, 'clear' to reset")
    print("-" * 70)

    turn=1

    while True:
        user_input=input(f"\nTurn {turn} You: ").strip()

        if user_input.lower()=='quit':
            print("Goodbye!!")
            break
        elif user_input.lower()=='state':
            print_full_state(state, turn)
            continue
        elif user_input.lower()=='clear':
            config={"configurable": {"thread_id": f"{user_id}_{turn}"}}
            state={"messages":[],"internal_summary":"","user_id":user_id}
            print("\n\nConversation cleared!!!.")
            continue

        state["messages"].append(HumanMessage(content=user_input))

        result=app.invoke(config=config, state=state)

        state.update(result)

        if state["messages"] and isinstance(state["messages"][-1], AIMessage):
            print(f"\n🤖 Assistant: {state['messages'][-1].content}")

        print_full_state(state, turn)

        turn+=1

# Test Suite
def run_tests():
    """Run predefined tests to demostrate the agent's capabilities."""
    print("🧪 Running Private Thought Agent Tests")
    print("=" * 70)

    app=create_private_thought_agent()
    
    test_cases=[{
        "user_id": "test_user_1",
            "messages": [
                "I'm really frustrated with this coding problem. Nothing works!",
                # "What is my name?",
                "I'm a complete beginner at Python, can you help?",
                "What is my experience level?"
            ]
    },
    {
            "user_id": "test_user_2", 
            "messages": [
                "I'm feeling great today! Just finished a big project.",
                "Tell me something interesting about AI.",
                "Actually, I'm getting tired now.",
                "Goodbye!"
            ]
        }
    ]

    for test_idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {test_idx}: User ID: {test_case['user_id']}")
        print('='*70)
        
        config = {"configurable": {"thread_id": test_case["user_id"]}}
        state = {"messages": [], "internal_summary": "", "user_id": test_case["user_id"]}
        
        for msg_idx, user_msg in enumerate(test_case["messages"], 1):
            print(f"\n💬 Turn {msg_idx}: User says: '{user_msg}'")
            print("-" * 50)
            
            # Add user message
            state["messages"].append(HumanMessage(content=user_msg))
            
            # Invoke agent
            result = app.invoke(state, config)
            state.update(result)
            
            # Print assistant response
            if state["messages"] and isinstance(state["messages"][-1], AIMessage):
                print(f"🤖 Assistant responds: {state['messages'][-1].content[:80]}...")
            
            # Print internal summary update
            print(f"\n🔒 Internal summary updated to:")
            print(textwrap.fill(state.get('internal_summary', 'No summary'), width=60))
            print("-" * 50)
    
    print("\n✅ All tests completed!")
    
    # Verify internal summary is never in user messages
    print("\n" + "="*70)
    print("VERIFICATION: Checking that internal_summary is NEVER shown to user")
    print("="*70)
    
    for test_case in test_cases:
        config = {"configurable": {"thread_id": test_case["user_id"]}}
        state = app.get_state(config)
        
        if state and state.values:
            internal_summary = state.values.get("internal_summary", "")
            messages = state.values.get("messages", [])
            
            print(f"\nUser: {test_case['user_id']}")
            print(f"Internal summary exists: {'Yes' if internal_summary else 'No'}")
            
            # Check that no message contains the internal summary
            for msg in messages:
                if isinstance(msg, AIMessage) and internal_summary:
                    if any(word in msg.content for word in internal_summary.split()[:5]):
                        print("❌ ERROR: Internal summary leaked to user!")
                    else:
                        print("✅ Internal summary successfully hidden from user")
            print()     
        

if __name__ == "__main__":
    print("=" * 70)
    print("PRIVATE THOUGHT AGENT DEMONSTRATION")
    print("=" * 70)
    print("\nThis agent maintains internal thoughts about the user's mood/experience")
    print("that are NEVER revealed in the conversation.")
    print("\nOptions:")
    print("1. Run tests to see the agent in action")
    print("2. Interactive chat mode")
    print("3. Use the production agent class")
    
    choice = input("\nChoose option (1, 2, or 3): ").strip()
    
    if choice == "1":
        run_tests()
    elif choice == "2":
        chat_with_agent()
    elif choice == "3":
        # Demo of production agent
        agent = PrivateAgent()
        
        # Simulate conversation
        conversations = [
            ("user_1", "I'm so angry! This code won't compile!"),
            ("user_1", "Can you help me fix it?"),
            ("user_2", "Hi, I'm new to programming"),
            ("user_2", "What's a variable?"),
            ("user_1", "Thanks, it works now!"),
        ]
        
        for user_id, message in conversations:
            print(f"\n{'='*50}")
            print(f"User {user_id}: {message}")
            result = agent.chat(user_id, message)
            print(f"Assistant: {result['response'][:100]}...")
            print(f"\nInternal thoughts for {user_id}:")
            print(textwrap.fill(result['internal_summary'], width=60))
    else:
        print("Invalid choice. Running tests by default...")
        run_tests()
        
