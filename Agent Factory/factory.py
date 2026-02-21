import operator
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send 

# --- 1. Schema ---
class SubTopicList(BaseModel):
    topics: List[str] = Field(description="3-5 sub-topics")

# --- 2. State Definition ---
class OverallState(TypedDict):
    topic: str
    # Initialize as an empty list to avoid KeyErrors
    topics: Annotated[List[str], operator.add] 
    summaries: Annotated[List[str], operator.add]

class WorkerState(TypedDict):
    sub_topic: str

# Use a faster model if possible, llama3.1:8b works great
llm = ChatOllama(model="llama3.1", temperature=0)
structured_llm = llm.with_structured_output(SubTopicList)

# --- 3. Node Logic ---
def planner(state: OverallState):
    print(f"--- Brainstorming topics for: {state['topic']} ---")
    
    # We use a very explicit prompt to force separation
    prompt = (
        f"You are a research planner. Break the topic '{state['topic']}' "
        "into 4 distinct chronological or thematic sub-topics. "
        "Respond ONLY with a JSON object in this format: "
        '{"topics": ["sub-topic 1", "sub-topic 2", "sub-topic 3", "sub-topic 4"]}'
    )
    
    # Using the structured LLM, but adding a 'force' prompt
    result = structured_llm.invoke(prompt)
    
    # Validation: If the model just returned the original topic, we manually split it
    if len(result.topics) <= 1:
        print("!!! Model failed to break down topic. Using fallback split. !!!")
        # Fallback logic if Ollama is being stubborn
        return {"topics": ["Formation", "Main Sequence", "Red Giant Phase", "Supernova/Remnant"]}
    
    return {"topics": result.topics}

def dispatch_researchers(state: OverallState):
    """
    This is the Router. It reads the 'topics' we just 
    saved to the state in the planner node.
    """
    topics = state.get("topics", [])
    if not topics:
        # Fallback if the LLM failed to return topics
        return [Send("aggregator", {"summaries": ["No topics found."] })]
        
    print(f"--- Dispatching {len(topics)} researchers ---")
    return [Send("researcher", {"sub_topic": t}) for t in topics]

def researcher(state: WorkerState):
    print(f"--- Researcher working on: {state['sub_topic']} ---")
    prompt = f"Write a 2-sentence expert summary about: {state['sub_topic']}"
    response = llm.invoke(prompt)
    return {"summaries": [f"[{state['sub_topic']}]: {response.content}"]}

def aggregator(state: OverallState):
    print("--- Finalizing Compendium ---")
    report = "\n\n".join(state["summaries"])
    return {"summaries": [f"FINAL REPORT:\n{report}"]}

# --- 4. Building the Graph ---
builder = StateGraph(OverallState)

builder.add_node("planner", planner)
builder.add_node("researcher", researcher)
builder.add_node("aggregator", aggregator)

builder.add_edge(START, "planner")

# Conditional edge handles the fan-out
builder.add_conditional_edges(
    "planner", 
    dispatch_researchers, 
    ["researcher", "aggregator"] # Possible destinations
)

# After each researcher finishes, they report to the aggregator
builder.add_edge("researcher", "aggregator")
builder.add_edge("aggregator", END)

graph = builder.compile()

# --- 5. Execution ---
if __name__ == "__main__":
    initial_state = {
        "topic": "The Lifecycle of a Star",
        "topics": [],    # Initialize explicitly
        "summaries": []  # Initialize explicitly
    }
    
    # Using 'stream' to see the progress
    for chunk in graph.stream(initial_state, stream_mode="updates"):
        print(chunk)