from typing import TypedDict, Optional
from langgraph.graph  import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

# Define State
class State(TypedDict):
    user_hobby: Optional[str]
    greeting_sent: bool

llm=ChatOllama(model="llama3.1")

# Define Nodes
def gatherer(state: State):
    if not state.get("user_hobby"):
        print("--- Gatherer: ")
        return {"greeting_sent": True}
    return {"greeting_sent": True}

def ask_human(state: State):
    # This node acts as a "parking lot"
    return {}

def final_response(state: State):
    hobby= state.get("user_hobby")
    return {"greeting_sent": True}

# Build Graph
workflow=StateGraph(State)

workflow.add_node("gatherer", gatherer)
workflow.add_node("ask_human", ask_human)
workflow.add_node("final_response", final_response)

workflow.add_edge(START, "gatherer")
workflow.add_conditional_edges(
    "gatherer",
    lambda state: "ask_human" if not state.get("user_hobby") else "final_response"

)
workflow.add_edge("ask_human","gatherer")
workflow.add_edge("final_response",END)

memory = MemorySaver()
app=workflow.compile(checkpointer=memory, interrupt_after=["ask_human"])


config = {"configurable": {"thread_id": "session_123"}}

# The graph will run 'gatherer' -> 'ask_human' and then PAUSE.
print("Starting Graph...")
for event in app.stream({"user_hobby": None}, config):
    print(event)

print("\n--- Human is typing... ---")
# We pretend the user said "Rock Climbing"
app.update_state(config, {"user_hobby": "Rock Climbing"}, as_node="ask_human")

print("\nResuming Graph...")
# Passing None tells LangGraph to look at the checkpoint and continue
for event in app.stream(None, config):
    print(event)