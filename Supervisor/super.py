import os
from typing import Literal
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Define the Structured Output Schema
class Router(BaseModel):
    """Decide which worker to call next or finish the task."""
    next_step: Literal["Searcher", "Coder", "FINISH"] = Field(
        description="The next worker to invoke. If the task is complete, use FINISH."
    )

# 2. Initialize the Local LLM
# Ensure you have 'ollama run llama3.1' active
llm = ChatOllama(model="llama3.1")

# --- Worker A: Searcher ---
def searcher_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    print("--- EXECUTING SEARCHER ---")
    # We use a specific tag 'SEARCHER_COMPLETED' so the Supervisor can find it easily
    result = "SEARCHER_COMPLETED: The current price of Bitcoin (BTC) is $98,450.20."
    return Command(
        update={"messages": [AIMessage(content=result, name="Searcher")]},
        goto="supervisor"
    )

# --- Worker B: Coder ---
def coder_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    print("--- EXECUTING CODER ---")
    # We use a specific tag 'CODER_COMPLETED'
    result = """CODER_COMPLETED: Here is the Python script:
```python
btc_price = 98450.20
investment = 1000
btc_owned = investment / btc_price
print(f'1000 USD is worth {btc_owned:.6f} BTC')
```"""
    return Command(
        update={"messages": [AIMessage(content=result, name="Coder")]},
        goto="supervisor"
    )

# --- The Supervisor (The Manager) ---
def supervisor_node(state: MessagesState) -> Command[Literal["Searcher", "Coder", "__end__"]]:
    structured_llm = llm.with_structured_output(Router)
    
    # We provide a strict checklist for the local model to follow
    system_prompt = (
        "You are the Manager of a team. Your job is to check the conversation history and decide the next step:\n"
        "1. If 'SEARCHER_COMPLETED' is NOT in the history, return 'Searcher'.\n"
        "2. If 'CODER_COMPLETED' is NOT in the history, return 'Coder'.\n"
        "3. If BOTH 'SEARCHER_COMPLETED' and 'CODER_COMPLETED' are present, return 'FINISH'.\n"
        "LOOK CAREFULLY at the messages before deciding. Do not repeat work."
    )
    
    # Combine system prompt with message history
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Get prediction
    prediction = structured_llm.invoke(messages)
    
    print(f"--- SUPERVISOR DECISION: {prediction.next_step} ---")
    
    if prediction.next_step == "FINISH":
        return Command(goto=END)
    
    return Command(goto=prediction.next_step)

# 3. Build the Graph
builder = StateGraph(MessagesState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("Searcher", searcher_node)
builder.add_node("Coder", coder_node)

builder.add_edge(START, "supervisor")

# 4. Compile the Graph
graph = builder.compile()

# 5. Run the Graph
if __name__ == "__main__":
    inputs = {
        "messages": [
            HumanMessage(content="Find the current price of Bitcoin and write a Python script to calculate how much $1000 would be worth")
        ]
    }
    
    # We use a recursion_limit of 10 to prevent runaway loops while testing
    try:
        result = graph.invoke(inputs, config={"recursion_limit": 10})
        print("\n--- FINAL GRAPH OUTPUT ---")
        for msg in result["messages"]:
            print(f"{msg.name if hasattr(msg, 'name') else 'User'}: {msg.content[:100]}...")
    except Exception as e:
        print(f"\nGraph stopped: {e}")