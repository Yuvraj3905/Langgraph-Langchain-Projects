from typing import TypedDict, List, Annotated, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
import json
import time

# ============== 1. Define Sub-Graph State Schema ==============
class WriterState(TypedDict):
    """State for the Writer sub-graph (specialized agent)."""
    draft_content: str  # Draft text
    edited_content: str  # Edited/professional version
    topic: str  # Writing topic
    style_guide: str  # Writing style guidelines
    draft_feedback: str  # Feedback on draft
    word_count: int  # Current word count
    is_finalized: bool  # Whether editing is complete

# ============== 2. Define Parent Graph State Schema ==============
class ManagerState(TypedDict):
    """State for the Manager parent graph."""
    messages: Annotated[List[BaseMessage], add_messages]  # Conversation
    user_intent: str  # Detected user intent
    requires_writer: bool  # Whether writer is needed
    writing_task: Optional[Dict[str, Any]]  # Task details for writer
    writer_result: Optional[str]  # Result from writer sub-graph
    general_response: Optional[str]  # Response for general chat
    user_profile: Dict[str, Any]  # User information
    conversation_history: List[str]  # Summary of conversation

# ============== 3. Helper: State Mapping Functions ==============
class StateMapper:
    """Handles state conversion between parent and child graphs."""
    
    @staticmethod
    def parent_to_writer(parent_state: ManagerState) -> WriterState:
        """
        Extract only necessary data from parent state for writer sub-graph.
        This prevents passing unnecessary data to the specialized agent.
        """
        # Get the writing task details
        writing_task = parent_state.get("writing_task", {})
        
        # Extract only what the writer needs
        return WriterState(
            draft_content="",
            edited_content="",
            topic=writing_task.get("topic", "Unknown topic"),
            style_guide=writing_task.get("style_guide", "Professional, clear, engaging"),
            draft_feedback="",
            word_count=0,
            is_finalized=False
        )
    
    @staticmethod
    def writer_to_parent(writer_state: WriterState, parent_state: ManagerState) -> ManagerState:
        """
        Merge writer results back into parent state.
        Preserves parent data while incorporating writer output.
        """
        # Create a copy of parent state
        updated_state = dict(parent_state)
        
        # Add writer result
        updated_state["writer_result"] = writer_state.get("edited_content", writer_state.get("draft_content", ""))
        
        # Update writing task with completion status
        if "writing_task" in updated_state:
            updated_state["writing_task"]["completed"] = True
            updated_state["writing_task"]["word_count"] = writer_state.get("word_count", 0)
            updated_state["writing_task"]["is_finalized"] = writer_state.get("is_finalized", False)
        
        return updated_state
    
    @staticmethod
    def extract_writing_task(parent_state: ManagerState) -> Dict[str, Any]:
        """Extract writing task from parent state messages."""
        # Get latest user message
        user_message = ""
        for msg in reversed(parent_state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        # Parse writing task from message
        task = {
            "topic": user_message,
            "style_guide": "Professional, engaging, informative",
            "audience": "General audience",
            "tone": "Neutral",
            "length": "Medium"
        }
        
        # Enhance with user profile if available
        user_profile = parent_state.get("user_profile", {})
        if user_profile.get("preferred_style"):
            task["style_guide"] = user_profile["preferred_style"]
        
        return task

# ============== 4. Build the Writer Sub-Graph ==============
def build_writer_subgraph():
    """Build and compile the writer sub-graph."""
    
    # Initialize LLM for writing tasks
    writer_llm = ChatOllama(
        model="llama3.1",
        temperature=0.8,  # Higher temp for creative writing
        num_predict=512,
    )
    
    # Initialize LLM for editing tasks
    editor_llm = ChatOllama(
        model="llama3.1",
        temperature=0.3,  # Lower temp for precise editing
        num_predict=256,
    )
    
    # Node 1: Draft Node
    def draft_node(state: WriterState) -> Dict[str, Any]:
        """Create initial draft based on topic."""
        
        print("\n" + "="*60)
        print("📝 WRITER SUB-GRAPH: DRAFT NODE")
        print("="*60)
        print(f"Topic: {state['topic']}")
        print(f"Style: {state['style_guide']}")
        
        # Create drafting prompt
        draft_prompt = f"""
        Write a draft paragraph about: {state['topic']}
        
        Style guidelines: {state['style_guide']}
        
        Requirements:
        - Be engaging and informative
        - Include key points about the topic
        - Maintain a professional tone
        - Target word count: 150-200 words
        
        Write only the draft content, no explanations.
        """
        
        # Generate draft
        draft_response = writer_llm.invoke([HumanMessage(content=draft_prompt)])
        draft_content = draft_response.content.strip()
        
        # Calculate word count
        word_count = len(draft_content.split())
        
        print(f"Draft generated ({word_count} words)")
        print("-" * 40)
        print(draft_content[:200] + "..." if len(draft_content) > 200 else draft_content)
        print("="*60)
        
        return {
            "draft_content": draft_content,
            "word_count": word_count,
            "draft_feedback": "Initial draft created"
        }
    
    # Node 2: Edit Node
    def edit_node(state: WriterState) -> Dict[str, Any]:
        """Edit draft to make it more professional."""
        
        print("\n" + "="*60)
        print("✏️  WRITER SUB-GRAPH: EDIT NODE")
        print("="*60)
        print(f"Editing draft ({state.get('word_count', 0)} words)")
        
        # Create editing prompt
        edit_prompt = f"""
        Edit this draft to make it more professional and polished:
        
        DRAFT:
        {state.get('draft_content', '')}
        
        Style guidelines: {state['style_guide']}
        
        Editing tasks:
        1. Improve clarity and flow
        2. Fix any grammatical errors
        3. Enhance vocabulary
        4. Ensure professional tone
        5. Maintain original meaning
        
        Provide only the edited version, no explanations.
        """
        
        # Generate edited version
        edit_response = editor_llm.invoke([HumanMessage(content=edit_prompt)])
        edited_content = edit_response.content.strip()
        
        # Generate feedback on changes
        feedback_prompt = f"""
        Compare the original draft with the edited version:
        
        ORIGINAL: {state.get('draft_content', '')[:200]}...
        EDITED: {edited_content[:200]}...
        
        Provide brief feedback on what was improved (2-3 sentences).
        """
        
        feedback_response = editor_llm.invoke([HumanMessage(content=feedback_prompt)])
        feedback = feedback_response.content.strip()
        
        print("Editing completed")
        print("-" * 40)
        print("Feedback:", feedback)
        print("-" * 40)
        print("Edited content preview:", edited_content[:200] + "..." if len(edited_content) > 200 else edited_content)
        print("="*60)
        
        return {
            "edited_content": edited_content,
            "draft_feedback": feedback,
            "is_finalized": True
        }
    
    # Build the writer sub-graph
    writer_workflow = StateGraph(WriterState)
    
    writer_workflow.add_node("draft", draft_node)
    writer_workflow.add_node("edit", edit_node)
    
    writer_workflow.add_edge("draft", "edit")
    writer_workflow.add_edge("edit", END)
    
    writer_workflow.set_entry_point("draft")
    
    # Compile the sub-graph
    writer_app = writer_workflow.compile()
    
    return writer_app

# ============== 5. Build the Parent Graph (Manager) ==============
def build_parent_graph(writer_subgraph):
    """Build the parent graph that delegates to writer sub-graph."""
    
    # Initialize LLM for intent classification
    classifier_llm = ChatOllama(
        model="llama3.1",
        temperature=0.1,  # Low temp for consistent classification
        num_predict=128,
    )
    
    # Node 1: Router Node
    def router_node(state: ManagerState) -> Dict[str, Any]:
        """Analyze user intent and decide whether writer is needed."""
        
        print("\n" + "="*70)
        print("🔄 PARENT GRAPH: ROUTER NODE")
        print("="*70)
        
        # Get latest user message
        user_message = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        # Classify intent
        classify_prompt = f"""
        Analyze the user's message and determine if they need a professional writer.
        
        User message: "{user_message}"
        
        Classification categories:
        1. WRITING_NEEDED - If the user explicitly asks to write, draft, create content, blog post, article, essay, report, etc.
        2. GENERAL_CHAT - If it's a conversation, question, or doesn't require professional writing
        
        Respond in JSON format:
        {{
            "intent": "WRITING_NEEDED" or "GENERAL_CHAT",
            "confidence": 0.0 to 1.0,
            "reason": "Brief explanation",
            "writing_task": {{
                "type": "blog_post" or "article" or "essay" or "report" or "other",
                "topic": "Extracted topic from message"
            }}  # Only if intent is WRITING_NEEDED
        }}
        """
        
        classification_response = classifier_llm.invoke([HumanMessage(content=classify_prompt)])
        
        try:
            # Parse JSON response
            classification_text = classification_response.content.strip()
            json_start = classification_text.find('{')
            json_end = classification_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                classification = json.loads(classification_text[json_start:json_end])
            else:
                classification = {
                    "intent": "GENERAL_CHAT",
                    "confidence": 0.5,
                    "reason": "Could not parse classification"
                }
        except json.JSONDecodeError:
            classification = {
                "intent": "GENERAL_CHAT",
                "confidence": 0.5,
                "reason": "Invalid JSON response"
            }
        
        # Determine if writer is needed
        requires_writer = classification.get("intent") == "WRITING_NEEDED"
        
        print(f"User message: {user_message[:100]}...")
        print(f"Detected intent: {classification.get('intent')}")
        print(f"Confidence: {classification.get('confidence')}")
        print(f"Reason: {classification.get('reason')}")
        
        if requires_writer:
            # Extract writing task
            writing_task = classification.get("writing_task", {})
            if not writing_task:
                writing_task = {"topic": user_message, "type": "blog_post"}
            
            # Add style guide based on task type
            style_guides = {
                "blog_post": "Engaging, conversational, include subheadings",
                "article": "Informative, well-structured, include references",
                "essay": "Academic, thesis-driven, include arguments",
                "report": "Formal, data-driven, include executive summary"
            }
            
            writing_task["style_guide"] = style_guides.get(
                writing_task.get("type", "blog_post"),
                "Professional, clear, engaging"
            )
            
            print(f"Writing task: {writing_task}")
            print("Decision: Route to WRITING_DEPT")
        else:
            print("Decision: Route to GENERAL_CHAT")
        
        print("="*70)
        
        return {
            "user_intent": classification.get("intent", "GENERAL_CHAT"),
            "requires_writer": requires_writer,
            "writing_task": writing_task if requires_writer else None
        }
    
    # Node 2: Writing Department Node (invokes sub-graph)
    def writing_dept_node(state: ManagerState) -> Dict[str, Any]:
        """Invoke the writer sub-graph with appropriate state."""
        
        print("\n" + "="*70)
        print("🏢 PARENT GRAPH: WRITING_DEPT NODE")
        print("="*70)
        print("Invoking Writer Sub-graph...")
        
        # Extract necessary state for writer
        writer_state = StateMapper.parent_to_writer(state)
        
        print(f"Writer sub-graph input:")
        print(f"  Topic: {writer_state['topic']}")
        print(f"  Style: {writer_state['style_guide']}")
        
        # Invoke the writer sub-graph
        try:
            writer_result = writer_subgraph.invoke(writer_state)
            
            print("\n✅ Writer sub-graph completed successfully!")
            print(f"Word count: {writer_result.get('word_count', 0)}")
            print(f"Is finalized: {writer_result.get('is_finalized', False)}")
            print("="*70)
            
            # Merge results back to parent state
            updated_state = StateMapper.writer_to_parent(writer_result, state)
            
            # Create final response message
            final_content = writer_result.get("edited_content", writer_result.get("draft_content", ""))
            response_message = f"I've written a professional piece about '{writer_state['topic']}':\n\n{final_content}"
            
            return {
                "writer_result": final_content,
                "messages": state["messages"] + [AIMessage(content=response_message)],
                **{k: v for k, v in updated_state.items() if k not in ["messages", "writer_result"]}
            }
            
        except Exception as e:
            print(f"❌ Error in writer sub-graph: {e}")
            print("="*70)
            
            # Fallback response
            error_message = f"I encountered an issue while writing about '{writer_state['topic']}'. Let me provide a brief response instead..."
            
            return {
                "writer_result": f"Error: {e}",
                "messages": state["messages"] + [AIMessage(content=error_message)]
            }
    
    # Node 3: General Chat Node
    def general_chat_node(state: ManagerState) -> Dict[str, Any]:
        """Handle general conversation without writer."""
        
        print("\n" + "="*70)
        print("💬 PARENT GRAPH: GENERAL_CHAT NODE")
        print("="*70)
        
        # Initialize chat LLM
        chat_llm = ChatOllama(
            model="llama3.1",
            temperature=0.7,
            num_predict=256,
        )
        
        # Get conversation history
        messages = state.get("messages", [])
        
        # Generate response
        response = chat_llm.invoke(messages[-3:])  # Last 3 messages for context
        
        print(f"Generated general response")
        print("-" * 40)
        print(response.content[:200] + "..." if len(response.content) > 200 else response.content)
        print("="*70)
        
        return {
            "general_response": response.content,
            "messages": state["messages"] + [response]
        }
    
    # Node 4: Response Formatter Node
    def response_formatter_node(state: ManagerState) -> Dict[str, Any]:
        """Format final response based on which path was taken."""
        
        print("\n" + "="*70)
        print("📄 PARENT GRAPH: RESPONSE_FORMATTER NODE")
        print("="*70)
        
        if state.get("requires_writer"):
            # Use writer result
            final_response = state.get("writer_result", "No writing result available")
            source = "Writer Department"
        else:
            # Use general chat result
            final_response = state.get("general_response", "No response available")
            source = "General Chat"
        
        # Format the response nicely
        formatted_response = f"🤖 **Response from {source}:**\n\n{final_response}"
        
        print(f"Formatting response from: {source}")
        print(f"Response length: {len(final_response)} characters")
        print("="*70)
        
        return {
            "messages": [AIMessage(content=formatted_response)]
        }
    
    # Custom Router Function
    def parent_router(state: ManagerState) -> str:
        """Route to appropriate node based on intent."""
        
        requires_writer = state.get("requires_writer", False)
        
        print("\n" + "="*70)
        print("🚦 PARENT GRAPH ROUTER - Making Decision")
        print("="*70)
        print(f"Requires writer: {requires_writer}")
        
        if requires_writer:
            print("Routing to: WRITING_DEPT")
            return "writing_dept"
        else:
            print("Routing to: GENERAL_CHAT")
            return "general_chat"
    
    # Build the parent graph
    parent_workflow = StateGraph(ManagerState)
    
    # Add all nodes
    parent_workflow.add_node("router", router_node)
    parent_workflow.add_node("writing_dept", writing_dept_node)
    parent_workflow.add_node("general_chat", general_chat_node)
    parent_workflow.add_node("formatter", response_formatter_node)
    
    # Set entry point
    parent_workflow.set_entry_point("router")
    
    # Add conditional routing
    parent_workflow.add_conditional_edges(
        "router",
        parent_router,
        {
            "writing_dept": "writing_dept",
            "general_chat": "general_chat"
        }
    )
    
    # Add edges to formatter
    parent_workflow.add_edge("writing_dept", "formatter")
    parent_workflow.add_edge("general_chat", "formatter")
    
    # Add edge from formatter to END
    parent_workflow.add_edge("formatter", END)
    
    # Compile the parent graph
    parent_app = parent_workflow.compile()
    
    return parent_app

# ============== 6. Test Functions ==============
def test_parent_child_integration():
    """Test the parent-child graph integration."""
    
    print("🧪 TESTING PARENT-CHILD GRAPH ARCHITECTURE")
    print("=" * 70)
    
    # Build the writer sub-graph
    print("\n1. Building Writer Sub-graph...")
    writer_app = build_writer_subgraph()
    print("✅ Writer sub-graph compiled successfully!")
    
    # Build the parent graph with sub-graph
    print("\n2. Building Parent Graph with Writer integration...")
    parent_app = build_parent_graph(writer_app)
    print("✅ Parent graph compiled successfully!")
    
    # Test Case 1: Writing request
    print("\n" + "="*70)
    print("📝 TEST CASE 1: Writing Request")
    print("="*70)
    print("User: 'Write a blog post about AI in healthcare'")
    
    initial_state_1 = {
        "messages": [HumanMessage(content="Write a blog post about AI in healthcare")],
        "user_intent": "",
        "requires_writer": False,
        "writing_task": None,
        "writer_result": None,
        "general_response": None,
        "user_profile": {"preferred_style": "Professional with real-world examples"},
        "conversation_history": []
    }
    
    print("\n🚀 Executing parent graph...")
    result_1 = parent_app.invoke(initial_state_1)
    
    print("\n✅ Test 1 completed!")
    print(f"Final response preview: {result_1['messages'][-1].content[:200]}...")
    
    # Test Case 2: General chat
    print("\n\n" + "="*70)
    print("💬 TEST CASE 2: General Chat")
    print("="*70)
    print("User: 'Hello, how are you today?'")
    
    initial_state_2 = {
        "messages": [HumanMessage(content="Hello, how are you today?")],
        "user_intent": "",
        "requires_writer": False,
        "writing_task": None,
        "writer_result": None,
        "general_response": None,
        "user_profile": {},
        "conversation_history": []
    }
    
    print("\n🚀 Executing parent graph...")
    result_2 = parent_app.invoke(initial_state_2)
    
    print("\n✅ Test 2 completed!")
    print(f"Final response preview: {result_2['messages'][-1].content[:200]}...")
    
    # Test Case 3: Mixed conversation
    print("\n\n" + "="*70)
    print("🔄 TEST CASE 3: Mixed Conversation")
    print("="*70)
    
    # Start with general chat
    state = {
        "messages": [HumanMessage(content="Hi there!")],
        "user_intent": "",
        "requires_writer": False,
        "writing_task": None,
        "writer_result": None,
        "general_response": None,
        "user_profile": {},
        "conversation_history": []
    }
    
    print("User: 'Hi there!'")
    result_3a = parent_app.invoke(state)
    print(f"Response: {result_3a['messages'][-1].content[:100]}...")
    
    # Then ask for writing
    print("\nUser: 'Now write an article about climate change'")
    state = result_3a
    state["messages"].append(HumanMessage(content="Now write an article about climate change"))
    result_3b = parent_app.invoke(state)
    
    print("\n✅ Test 3 completed!")
    print(f"Final response preview: {result_3b['messages'][-1].content[:200]}...")
    
    return parent_app, writer_app

# ============== 7. Advanced: State Passing Study ==============
def study_state_passing():
    """
    Deep dive study on state passing between graphs with different schemas.
    """
    
    print("\n" + "="*70)
    print("🧠 DEEP DIVE: State Passing Between Graphs")
    print("="*70)
    
    # Example 1: Naive Approach (Passing everything)
    print("\n1. ❌ NAIVE APPROACH: Passing entire state")
    print("-" * 70)
    
    class ParentStateNaive(TypedDict):
        messages: List[BaseMessage]
        user_data: Dict[str, Any]
        session_data: Dict[str, Any]
        analytics: Dict[str, Any]
        config: Dict[str, Any]
        writing_task: Dict[str, Any]
    
    class ChildStateNaive(TypedDict):
        # Child gets everything - inefficient!
        messages: List[BaseMessage]
        user_data: Dict[str, Any]
        session_data: Dict[str, Any]
        analytics: Dict[str, Any]
        config: Dict[str, Any]
        writing_task: Dict[str, Any]
        child_specific: str  # Only this is actually needed
    
    print("Problem: Child gets 6 fields but only needs 1")
    print("Inefficiency: 83% of data is irrelevant")
    print("Risk: LLM might get confused by irrelevant context")
    
    # Example 2: Efficient Approach (Selective passing)
    print("\n\n2. ✅ EFFICIENT APPROACH: Selective state passing")
    print("-" * 70)
    
    class ParentStateEfficient(TypedDict):
        messages: List[BaseMessage]
        user_data: Dict[str, Any]  # Large user profile
        session_data: Dict[str, Any]  # Session tracking
        analytics: Dict[str, Any]  # Usage analytics
        config: Dict[str, Any]  # System configuration
        writing_task: Dict[str, Any]  # Task details
    
    class ChildStateEfficient(TypedDict):
        # Child only gets what it needs
        task_details: Dict[str, Any]  # Extracted from writing_task
        user_context: Dict[str, Any]  # Minimal relevant user info
        child_specific: str  # Child's own data
    
    print("Solution: Extract only relevant data")
    print("Efficiency: ~90% data reduction")
    print("Benefits:")
    print("  - Faster execution")
    print("  - Cleaner context for LLM")
    print("  - Better separation of concerns")
    print("  - Easier debugging")
    
    # Example 3: Practical Implementation
    print("\n\n3. 🛠️ PRACTICAL IMPLEMENTATION")
    print("-" * 70)
    
    # Simulate large parent state
    simulated_parent_state = {
        "messages": [HumanMessage(content="Write about AI")] * 10,  # Large history
        "user_data": {
            "name": "John",
            "email": "john@example.com",
            "preferences": {"theme": "dark", "language": "en"},
            "history": ["query1", "query2", "query3"] * 10,
            "subscription": "premium",
            "devices": ["phone", "laptop", "tablet"]
        },
        "session_data": {
            "start_time": "2024-01-01T10:00:00",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0...",
            "page_views": 42,
            "interactions": ["click", "scroll", "type"] * 5
        },
        "analytics": {
            "total_queries": 150,
            "avg_response_time": 2.5,
            "success_rate": 0.95,
            "popular_topics": ["AI", "Science", "Technology"] * 3
        },
        "config": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful assistant..."
        },
        "writing_task": {
            "topic": "Artificial Intelligence",
            "style": "Professional",
            "audience": "General",
            "length": "Medium"
        }
    }
    
    # Extract only what child needs
    def extract_for_child(parent_state):
        """Efficient extraction of relevant data."""
        return {
            "task_details": parent_state.get("writing_task", {}),
            "user_context": {
                "preferred_style": parent_state.get("user_data", {}).get("preferences", {}).get("theme", "default")
            },
            "child_specific": "Writer agent configuration"
        }
    
    child_input = extract_for_child(simulated_parent_state)
    
    print(f"Parent state size: {len(str(simulated_parent_state))} characters")
    print(f"Child input size: {len(str(child_input))} characters")
    print(f"Reduction: {(1 - len(str(child_input))/len(str(simulated_parent_state)))*100:.1f}%")
    
    # Example 4: State Mapping Patterns
    print("\n\n4. 📊 STATE MAPPING PATTERNS")
    print("-" * 70)
    
    print("Pattern 1: Extract-Transform-Load (ETL)")
    print("  Parent → [Extract] → [Transform] → [Load] → Child")
    
    print("\nPattern 2: Facade Pattern")
    print("  Child only sees a simplified interface")
    
    print("\nPattern 3: Adapter Pattern")
    print("  Convert between different state schemas")
    
    print("\nPattern 4: Builder Pattern")
    print("  Incrementally build child state from parent")
    
    # Example 5: Real-world scenario
    print("\n\n5. 🌍 REAL-WORLD SCENARIO: Multi-agent System")
    print("-" * 70)
    
    print("Parent Agent Responsibilities:")
    print("  - User authentication")
    print("  - Session management")
    print("  - Analytics collection")
    print("  - Resource allocation")
    print("  - Intent classification")
    
    print("\nChild Agent (Writer) Responsibilities:")
    print("  - Content generation")
    print("  - Style adherence")
    print("  - Quality assurance")
    print("  - Formatting")
    
    print("\nKey Insight:")
    print("The writer doesn't need to know:")
    print("  - User's IP address")
    print("  - Session duration")
    print("  - Analytics data")
    print("  - System configuration")
    print("\nIt only needs:")
    print("  - Topic to write about")
    print("  - Style guidelines")
    print("  - Audience information")

# ============== 8. Interactive Demo ==============
def interactive_demo():
    """Interactive demo of the parent-child graph system."""
    
    print("🤖 INTERACTIVE PARENT-CHILD GRAPH DEMO")
    print("=" * 70)
    
    # Build both graphs
    print("Building graphs...")
    writer_app = build_writer_subgraph()
    parent_app = build_parent_graph(writer_app)
    
    print("\nGraphs built successfully!")
    print("\nYou can now chat with the system.")
    print("Type 'quit' to exit, 'state' to see current state")
    print("=" * 70)
    
    # Initial state
    state = {
        "messages": [],
        "user_intent": "",
        "requires_writer": False,
        "writing_task": None,
        "writer_result": None,
        "general_response": None,
        "user_profile": {"preferred_style": "Professional"},
        "conversation_history": []
    }
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'state':
            print("\n📊 CURRENT STATE:")
            print(f"Messages: {len(state.get('messages', []))}")
            print(f"User intent: {state.get('user_intent', 'None')}")
            print(f"Requires writer: {state.get('requires_writer', False)}")
            print(f"Writing task: {state.get('writing_task', 'None')}")
            print(f"Writer result: {'Exists' if state.get('writer_result') else 'None'}")
            continue
        
        # Add user message
        state["messages"].append(HumanMessage(content=user_input))
        
        # Invoke parent graph
        result = parent_app.invoke(state)
        
        # Update state
        state.update(result)
        
        # Display response
        if state["messages"] and isinstance(state["messages"][-1], AIMessage):
            print(f"\nAssistant: {state['messages'][-1].content[:300]}...")

# ============== 9. Main Execution ==============
if __name__ == "__main__":
    print("=" * 70)
    print("🏢 PARENT-CHILD GRAPH ARCHITECTURE")
    print("=" * 70)
    
    print("\nOptions:")
    print("1. Run integration tests")
    print("2. Study state passing (Deep Dive)")
    print("3. Interactive demo")
    print("4. Comprehensive demonstration")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        parent_app, writer_app = test_parent_child_integration()
        print("\n" + "="*70)
        print("✅ INTEGRATION TESTS COMPLETED")
        print("="*70)
        
    elif choice == "2":
        study_state_passing()
        
    elif choice == "3":
        interactive_demo()
        
    elif choice == "4":
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE DEMONSTRATION")
        print("="*70)
        
        # Part 1: Build and test
        print("\n📊 PART 1: Building and Testing Integration")
        print("-" * 70)
        parent_app, writer_app = test_parent_child_integration()
        
        # Part 2: State passing study
        print("\n\n📊 PART 2: State Passing Deep Dive")
        print("-" * 70)
        study_state_passing()
        
        # Part 3: Quick interactive test
        print("\n\n📊 PART 3: Quick Test")
        print("-" * 70)
        
        test_queries = [
            "Write a professional email about the quarterly report",
            "How's the weather today?",
            "Draft a blog post about renewable energy",
            "Can you help me with math homework?"
        ]
        
        for query in test_queries:
            print(f"\nUser: {query}")
            result = parent_app.invoke({
                "messages": [HumanMessage(content=query)],
                "user_intent": "",
                "requires_writer": False,
                "writing_task": None,
                "writer_result": None,
                "general_response": None,
                "user_profile": {},
                "conversation_history": []
            })
            
            if result.get("requires_writer"):
                print("✓ Used Writer sub-graph")
            else:
                print("✓ Used General chat")
            
            print(f"Response: {result['messages'][-1].content[:150]}...")
        
        print("\n" + "="*70)
        print("🎉 DEMONSTRATION COMPLETE")
        print("="*70)
        
    else:
        print("Running integration tests by default...")
        parent_app, writer_app = test_parent_child_integration()