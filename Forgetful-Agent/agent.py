from typing import List, TypedDict, Annotated, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
import operator

# Define the state typed dictionary
class AgentState(TypedDict):
    """State for our agent, holding conversation history."""
    messages: Annotated[List[BaseMessage],operator.add]
    thread_id: str


# Define the call_model node
def call_model(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """Node that calls a local ollama LLM and returns the response."""

    #Initialize local Ollama LLM
    llm=ChatOllama(
        model="llama3.1",
        temperature=0.7,
        num_predict=256,
    )

    #Call the LLM with the conversation history
    response=llm.invoke(state["messages"])

    #Return the AI's Response
    return {"messages":[response]}

# Build the Graph

def create_agent():
    """Create and compile the agent graph with Ollama."""

    #Create the graph
    workflow=StateGraph(AgentState)

    #Add the call_model node
    workflow.add_node("call_model",call_model)

    #Set the entry point
    workflow.set_entry_point("call_model")

    #Set the exit point
    workflow.add_edge("call_model",END)

    #Add persistence Layer(Memory Saver)
    memory=MemorySaver()

    app=workflow.compile(checkpointer=memory)

    return app

def save_graph_as_png(app, output_file="graph.png"):
    """Saves the graph as a PNG image."""
    try:
        # Generate the PNG bytes
        png_bytes = app.get_graph().draw_mermaid_png()
        with open(output_file, "wb") as f:
            f.write(png_bytes)
        print(f"Graph saved as {output_file}")
    except Exception as e:
        print(f"Error saving graph: {e}")


# Test the agent
def test_agent():
    """Test the agent by running it with specific scenarios."""
    app = create_agent()

    print("=" * 50)
    print("TEST 1: Send message to thread_id='1' saying 'My name is Yuvraj.'")
    print("=" * 50)
    
    # Test 1: Thread 1 - Tell the agent your name
    config_1 = {"configurable": {"thread_id": "1"}}
    
    # Send first message to thread 1
    result_1 = app.invoke(
        {"messages": [HumanMessage(content="My name is Yuvraj.")]},
        config_1
    )
    
    print("Human: My name is Yuvraj.")
    print(f"AI: {result_1['messages'][-1].content}")
    print()
    
    print("=" * 50)
    print("TEST 2: Send message to thread_id='2' asking 'What is my name?'")
    print("=" * 50)
    
    # Test 2: Thread 2 - Ask for name (shouldn't know)
    config_2 = {"configurable": {"thread_id": "2"}}
    
    result_2 = app.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config_2
    )
    
    print("Human: What is my name?")
    print(f"AI: {result_2['messages'][-1].content}")
    print()
    
    print("=" * 50)
    print("TEST 3: Send message back to thread_id='1' asking 'What is my name?'")
    print("=" * 50)
    
    # Test 3: Thread 1 - Ask for name again (should remember)
    result_3 = app.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config_1
    )
    
    print("Human: What is my name?")
    print(f"AI: {result_3['messages'][-1].content}")
    
    return app

if __name__ == "__main__":
    test_agent()
