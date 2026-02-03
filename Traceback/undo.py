from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

# Define State
class StoryState(TypedDict):
    story_parts: list[str]
    current_phase: str

llm = ChatOllama(model="llama3.1")

# Node Functions
def introduction(state: StoryState):
    res = llm.invoke("Write a 1-sentence intro about a space explorer named Kael.")
    return {"story_parts": [res.content], "current_phase": "conflict"}

def conflict(state: StoryState):
    # This will be branched later
    res = llm.invoke(f"Based on: {state['story_parts'][-1]}, write a 1-sentence conflict.")
    return {"story_parts": state['story_parts'] + [res.content], "current_phase": "conclusion"}

def conclusion(state: StoryState):
    res = llm.invoke(f"End this story in 1 sentence: {' '.join(state['story_parts'])}")
    return {"story_parts": state['story_parts'] + [res.content]}

# Build Graph
builder = StateGraph(StoryState)
builder.add_node("introduction", introduction)
builder.add_node("conflict", conflict)
builder.add_node("conclusion", conclusion)

builder.add_edge(START, "introduction")
builder.add_edge("introduction", "conflict")
builder.add_edge("conflict", "conclusion")
builder.add_edge("conclusion", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "thread_1"}}




# --- STEP 1: Run Timeline A ---
print("--- RUNNING TIMELINE A ---")
for event in app.stream({"story_parts": []}, config):
    print(event)

# --- STEP 2: List History & Find the Rewind Point ---
# We want the state AFTER 'introduction' but BEFORE 'conflict'
history = list(app.get_state_history(config))
# Usually, history[0] is the latest (Conclusion). 
# We look for the one where only the intro exists.
intro_checkpoint = next(s for s in history if s.metadata.get("step") == 1)
print(f"\nRewinding to Checkpoint ID: {intro_checkpoint.config['configurable']['checkpoint_id']}")

# --- STEP 3: The 1% Expert Move (The Fork) ---
# We update the state to inject a NEW conflict manually
new_conflict = {"story_parts": [intro_checkpoint.values['story_parts'][0], "Kael discovers the ship's AI has fallen in love with him."]}

app.update_state(
    intro_checkpoint.config, 
    new_conflict, 
    as_node="introduction" # We tell the graph we are 'at' the end of intro
)

# --- STEP 4: Run Timeline B ---
print("\n--- RUNNING TIMELINE B (The Rewritten Reality) ---")
# We resume from the new state
for event in app.stream(None, intro_checkpoint.config):
    print(event)