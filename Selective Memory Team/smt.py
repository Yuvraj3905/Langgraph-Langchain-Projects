from typing import TypedDict, List

class ParentState(TypedDict):
    user_input:str
    final_report: str

class ResearchState(TypedDict):
    topic: str
    search_queries: List[str]
    raw_links: List[str]
    draft_notes:str
    summary: str

def gather_research(state: ResearchState):
    print("--- [Sub-Graph] Researching Deeply ---")

    return {
        "search_queries": [f"{state['topic']} trends", f"{state['topic']} future"],
        "raw_links": ["https://tech.comp/123", "https://data.org/99"],
        "draft_notes": "Internal Though: This topic is highly volatile...."
    }

def summarize_findings(state: ResearchState):
    print("--- [Sub-Graph] Summarizing for Parent ---")
    return {"summary": f"Comprehensive report on {state['topic']}: All systems go."}

def call_researcher_dept(state: ParentState):
    """
    The Interface: Maps ParentState -> ResearchState
    and filters ResearchState -> ParentState.
    """
    child_input={"topic": state["user_input"]}

    # 2. Invoke Sub-graph (Simulated invocation)
    # In a real LangGraph, this would be: researcher_subgraph.invoke(child_input)
    # Let's simulate the full internal state the child generates:
    child_internal_result = {
        "topic": state["user_input"],
        "search_queries": ["query1"],
        "raw_links": ["http://secret-link.com"], # We want to lose this!
        "draft_notes": "Messy internal draft.",     # And this!
        "summary": "The final polished summary."
    }
    
    # 3. The Mapping Move: Return ONLY the summary to the ParentState
    return {"final_report": child_internal_result["summary"]}


# Initial State
current_parent_state: ParentState = {
    "user_input": "Quantum Computing",
    "final_report": ""
}

print("Initial Parent State:", current_parent_state)
print("-" * 30)

# Execute the Research Dept Node
update = call_researcher_dept(current_parent_state)
current_parent_state.update(update)

print("\nFinal Parent State (Post-Research):")
import json
print(json.dumps(current_parent_state, indent=4))

# Explicit check for leakage
if "raw_links" not in current_parent_state:
    print("\n✅ SUCCESS: 'raw_links' and 'draft_notes' were discarded.")
    print("The Parent State remains clean and executive-ready.")
