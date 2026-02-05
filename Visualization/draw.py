import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Define State
class AgentState(TypedDict):
    query: str
    results: List[str]
    reflection: str
    loop_count: int

# Define the nodes
def search_node(state: AgentState):
    print(f"---SEARCHING (ITERATION {state.get('loop_count',0)})---")
    new_data=f"Data point {state.get('loop_count',0)+1}"
    return {
        "results": state.get("results",[])+[new_data],
        "loop_count": state.get("loop_count",0)+1
    }

def reflect_node(state: AgentState):
    print("---REFLECTING---")

    quality_met =state["loop_count"]>=3
    return {"reflection": "Enough data gathered" if quality_met else "Need more info"}

workflow= StateGraph(AgentState)

workflow.add_node("Search", search_node)
workflow.add_node("Reflect", reflect_node)

workflow.set_entry_point("Search")
workflow.add_edge("Search", "Reflect")

workflow.add_conditional_edges(
    "Reflect",
    lambda state: "Final_Report" if state["loop_count"]>=3 else "Search",
    {
        "Search": "Search",
        "Final_Report": END
    }
)
app=workflow.compile()

from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as e:
    print(app.get_graph().draw_mermaid())

# Define the file path
output_path = "graph_structure.png"

try:
    # Get the PNG data (bytes) from the compiled app
    png_data = app.get_graph().draw_mermaid_png()
    
    # Write the bytes to a file in the current directory
    with open(output_path, "wb") as f:
        f.write(png_data)
        
    print(f"✅ Success! Graph saved to: {os.path.abspath(output_path)}")

except Exception as e:
    print(f"❌ Failed to save image. Error: {e}")
    print("Tip: Ensure you have 'pygraphviz' or 'mermaid.ink' access configured.")