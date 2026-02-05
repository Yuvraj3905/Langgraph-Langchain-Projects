import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama

# State Definiton 
class JournalistState(TypedDict):
    topic:str
    facts: List[str]
    article: str
    tone: str


llm = ChatOllama(
        model="llama3.1",
        temperature=0.7,  
        # num_predict=128,
    )

def research_node(state: JournalistState):
    print("--- RESEARCHING ---")
    # Simulating fact gathering
    return {"facts": [
        "Llama 3.1 is open-source.", 
        "It supports 405B parameters.", 
        "It was trained by Meta."
    ]}

def draft_node(state: JournalistState):
    print(f"--- DRAFTING (Tone: {state.get('tone', 'Professional')}) ---")
    prompt = f"Write a 2-sentence article about {state['topic']} using these facts: {state['facts']}. Use a {state.get('tone', 'Professional')} tone."
    response = llm.invoke(prompt)
    return {"article": response.content}

# 4. Build Graph with Interruption Points
workflow = StateGraph(JournalistState)
workflow.add_node("Research", research_node)
workflow.add_node("Draft", draft_node)

workflow.add_edge(START, "Research")
workflow.add_edge("Research", "Draft")
workflow.add_edge("Draft", END)

# EXPERT MOVE: Set interrupts before the approval nodes
memory = InMemorySaver()
app = workflow.compile(
    checkpointer=memory, 
    interrupt_before=["Draft"], # Interrupt 1: Review Facts
    interrupt_after=["Draft"]   # Interrupt 2: Review Article
)

config = {"configurable": {"thread_id": "journalist_01"}}

# Start the process
print("Starting initial run...")
app.invoke({"topic": "LangGraph", "tone": "Professional"}, config)

# PAUSED at "Draft" node. Let's look at the facts.
snapshot = app.get_state(config)
print(f"Current Facts: {snapshot.values['facts']}")

# MANUALLY PIVOT: Inject a new fact
app.update_state(config, {"facts": ["Llama 3.1 can now brew coffee (Fake Fact)"]})

# RESUME: The agent will now use the injected fact
print("\nResuming after pivot...")
app.invoke(None, config)

final_state = app.get_state(config)
print(f"Draft Result: {final_state.values['article']}")

# 1. Find the history of states
history = list(app.get_state_history(config))

# 2. Identify the checkpoint BEFORE the Draft node (usually index 1 or 2 in history)
# We want the state where 'facts' exist but 'article' doesn't yet.
pre_draft_state = [h for h in history if "facts" in h.values and "article" not in h.values][0]

print(f"Rewinding to Checkpoint: {pre_draft_state.config['configurable']['checkpoint_id']}")

# 3. Modify the state at that specific past checkpoint
# We change the tone to 'Sarcastic'
new_config = app.update_state(
    pre_draft_state.config, 
    {"tone": "Sarcastic"}, 
)

# 4. Resume from that specific fork in time
app.invoke(None, new_config)

rewound_state = app.get_state(config)
print(f"Sarcastic Rewind Result: {rewound_state.values['article']}")