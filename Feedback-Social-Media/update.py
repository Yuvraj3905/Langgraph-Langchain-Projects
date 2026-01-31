import operator
from typing import Annotated, TypedDict
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# 1. Define State
class AgentState(TypedDict):
    # Annotated with operator.add allows messages to accumulate
    messages: Annotated[list, operator.add]

# Initialize LLM (Ollama Llama 3.1)
llm = Ollama(model="llama3.1")

# 2. Define Nodes
def writer_node(state: AgentState):
    print("--- WRITER: Drafting post... ---")
    response = llm.invoke("Write a short social media post about AI in 2026.")
    return {"messages": [response]}

def publisher_node(state: AgentState):
    print("--- PUBLISHER: Final Step ---")
    # Get the last message in the state
    final_content = state["messages"][-1]
    print(f"\n🚀 PUBLISHED TO X: {final_content}\n")
    return state

# 3. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("Writer", writer_node)
workflow.add_node("Publisher", publisher_node)

workflow.add_edge(START, "Writer")
workflow.add_edge("Writer", "Publisher")
workflow.add_edge("Publisher", END)

# 4. Compile with Interruption and Memory
memory = InMemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_after=["Writer"]  # Pause specifically after Writer finishes
)

# --- EXECUTION FLOW ---

config = {"configurable": {"thread_id": "post_001"}}

# Step 1: Run until interrupt
print("Initial Run...")
app.invoke({"messages": []}, config)

# Step 2: Use get_state to see what was written
state = app.get_state(config)
original_draft = state.values["messages"][-1]
print(f"Captured Draft: {original_draft}")

# Step 3: THE "1% EXPERT" MOVE - Manual Update
# We replace the content by acting as the 'Writer' node
improved_draft = "AI in 2026 is no longer just a tool; it's our cognitive exoskeleton. 🤖✨ #AI2026 #Future"

print("\n--- HUMAN INTERVENTION: Improving the draft... ---")
app.update_state(
    config, 
    {"messages": [improved_draft]}, 
    as_node="Writer"
)

# Step 4: Resume
print("Resuming Graph...")
app.invoke(None, config)