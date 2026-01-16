from typing import TypedDict, List, Annotated, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from datetime import datetime
import time
import random
import uuid

# ============== 1. Define the State Schema ==============
class SelfCorrectingState(TypedDict):
    """State for the self-correcting agent."""
    messages: Annotated[List[BaseMessage], add_messages]  # Conversation history
    attempt_count: int  # Track how many times the loop has run
    search_query: str  # The original search query
    search_results: List[str]  # Accumulated search results
    last_search_quality: str  # "low" or "high"
    retry_hint: str  # Hint for retrying search
    recursion_depth: int  # Track recursion depth
    error_log: List[str]  # Log errors for debugging

# ============== 2. Searcher Node ==============
def searcher_node(state: SelfCorrectingState) -> Dict[str, Any]:
    """
    Node that simulates search behavior.
    Returns low quality for first 2 attempts, success on 3rd.
    """
    
    attempt = state["attempt_count"] + 1
    query = state.get("search_query", "unknown query")
    retry_hint = state.get("retry_hint", "")
    
    print("=" * 70)
    print(f"🔍 SEARCHER NODE - Attempt {attempt}")
    print("-" * 70)
    print(f"Query: {query}")
    print(f"Retry hint: {retry_hint}")
    
    # Simulate different search behaviors based on attempt count
    if attempt < 3:
        # First 2 attempts: low quality
        result = f"Low quality result: No data found for '{query}'. Try different keywords."
        quality = "low"
        print(f"Result: {result}")
    else:
        # 3rd+ attempt: success
        result = f"Success: Here is your data about '{query}': Found comprehensive information with detailed analysis."
        quality = "high"
        print(f"Result: {result}")
    
    print("=" * 70)
    
    # Simulate search latency
    time.sleep(0.5)
    
    # Update state
    current_results = state.get("search_results", [])
    current_results.append(result)
    
    return {
        "attempt_count": attempt,
        "search_results": current_results,
        "last_search_quality": quality,
        "messages": state["messages"] + [
            AIMessage(content=f"Search attempt {attempt}: {result}")
        ],
        "recursion_depth": state.get("recursion_depth", 0) + 1
    }

# ============== 3. Analyzer Node ==============
def analyzer_node(state: SelfCorrectingState) -> Dict[str, Any]:
    """
    Node that checks search quality and decides next step.
    If low quality, provides hint and routes back to searcher.
    If high quality, routes to END.
    """
    
    quality = state.get("last_search_quality", "low")
    attempt = state["attempt_count"]
    
    print("=" * 70)
    print(f"🔬 ANALYZER NODE - Checking quality")
    print("-" * 70)
    print(f"Attempt: {attempt}")
    print(f"Quality: {quality}")
    
    if quality == "low":
        # Generate a hint for the next search attempt
        query = state.get("search_query", "")
        hints = [
            f"Try searching with more specific terms related to '{query}'",
            f"Consider alternative spellings or synonyms for '{query}'",
            f"Add location or time constraints to '{query}'",
            f"Break down '{query}' into smaller sub-queries",
            f"Try academic or technical databases for '{query}'"
        ]
        
        hint = hints[min(attempt, len(hints) - 1)]  # Cycle through hints
        
        print(f"Quality too low. Providing hint: {hint}")
        print("Routing back to Searcher")
        print("=" * 70)
        
        return {
            "retry_hint": hint,
            "messages": state["messages"] + [
                SystemMessage(content=f"Analysis: Search quality low. Hint: {hint}")
            ],
            # Important: We DON'T return "should_retry" here
            # The routing decision happens in the conditional edge
        }
    else:
        # High quality - prepare final answer
        print("Quality acceptable. Preparing final answer...")
        
        final_answer = f"After {attempt} attempts, I found comprehensive information. Here's the summary of what I learned..."
        
        print(f"Final answer ready. Routing to END.")
        print("=" * 70)
        
        return {
            "messages": state["messages"] + [
                AIMessage(content=final_answer)
            ],
            # We'll route to END via conditional edge
        }

# ============== 4. Custom Routing Function ==============
def quality_router(state: SelfCorrectingState) -> str:
    """
    Custom router that decides next step based on search quality.
    """
    
    quality = state.get("last_search_quality", "low")
    attempt = state.get("attempt_count", 0)
    
    print("\n" + "=" * 70)
    print("🔄 QUALITY ROUTER - Making Decision")
    print("-" * 70)
    print(f"Attempt: {attempt}")
    print(f"Quality: {quality}")
    
    if quality == "low":
        if attempt >= 5:  # Safety limit
            print("⚠️  Maximum attempts reached! Raising error...")
            print("=" * 70)
            raise RecursionError(f"Maximum search attempts ({attempt}) exceeded")
        
        print("Routing back to Searcher")
        print("=" * 70)
        return "searcher"
    else:
        print("Routing to END (success)")
        print("=" * 70)
        return "__end__"

# ============== 5. Build the Graph with Recursion Limit ==============
def build_self_correcting_agent(use_checkpointer: bool = False):
    """
    Build the self-correcting agent with configurable recursion limit.
    """
    
    workflow = StateGraph(SelfCorrectingState)
    
    # Add nodes
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("analyzer", analyzer_node)
    
    # Set entry point
    workflow.set_entry_point("searcher")
    
    # Add edge from searcher to analyzer
    workflow.add_edge("searcher", "analyzer")
    
    # Add conditional edge from analyzer based on quality
    workflow.add_conditional_edges(
        "analyzer",
        quality_router,
        {
            "searcher": "searcher",
            "__end__": END
        }
    )
    
    # Compile with optional checkpointer
    if use_checkpointer:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
    else:
        app = workflow.compile()
    
    return app

# ============== 6. Test the Agent ==============
def test_self_correction():
    """Test the self-correcting agent."""
    
    print("🧪 TESTING SELF-CORRECTING AGENT")
    print("=" * 70)
    
    # Test 1: Normal success case (3 attempts)
    print("\n🔬 TEST 1: Normal flow (should succeed on 3rd attempt)")
    print("-" * 70)
    
    app = build_self_correcting_agent(use_checkpointer=False)  # Don't use checkpointer for simple test
    
    initial_state = {
        "messages": [HumanMessage(content="Find information about quantum computing")],
        "attempt_count": 0,
        "search_query": "quantum computing",
        "search_results": [],
        "last_search_quality": "low",
        "retry_hint": "",
        "recursion_depth": 0,
        "error_log": []
    }
    
    try:
        result = app.invoke(initial_state)
        print(f"\n✅ Success after {result['attempt_count']} attempts")
        print(f"Final message: {result['messages'][-1].content[:100]}...")
    except RecursionError as e:
        print(f"\n❌ Recursion error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
    
    # Test 2: Force recursion limit
    print("\n\n🔬 TEST 2: Force recursion limit (modify searcher to always fail)")
    print("-" * 70)
    
    def always_fail_searcher(state: SelfCorrectingState) -> Dict[str, Any]:
        """Modified searcher that always returns low quality."""
        attempt = state["attempt_count"] + 1
        
        return {
            "attempt_count": attempt,
            "last_search_quality": "low",
            "search_results": state.get("search_results", []) + [f"Failed attempt {attempt}"],
            "messages": state["messages"] + [
                AIMessage(content=f"Search attempt {attempt}: Failed")
            ],
            "recursion_depth": state.get("recursion_depth", 0) + 1
        }
    
    # Rebuild with always-failing searcher
    workflow = StateGraph(SelfCorrectingState)
    workflow.add_node("searcher", always_fail_searcher)
    workflow.add_node("analyzer", analyzer_node)
    workflow.set_entry_point("searcher")
    workflow.add_edge("searcher", "analyzer")
    workflow.add_conditional_edges(
        "analyzer",
        quality_router,
        {"searcher": "searcher", "__end__": END}
    )
    
    app = workflow.compile()
    
    try:
        result = app.invoke(initial_state)
        print(f"\nResult: {result}")
    except RecursionError as e:
        print(f"\n✅ Correctly raised RecursionError: {e}")
    except Exception as e:
        print(f"\n❌ Different error: {type(e).__name__}: {e}")

# ============== 7. DEEP DIVE: Max Concurrency in LangGraph ==============
def study_max_concurrency():
    """
    Study how max_concurrency works in LangGraph.
    Demonstrates parallel node execution and error isolation.
    """
    
    print("\n" + "=" * 70)
    print("🧠 DEEP DIVE: Max Concurrency in LangGraph")
    print("=" * 70)
    
    # Define a state for concurrent processing
    class ConcurrentState(TypedDict):
        agent_a_results: List[str]
        agent_b_results: List[str]
        agent_c_results: List[str]
        errors: List[str]
        execution_times: Dict[str, float]
    
    # Define three agents with different behaviors
    def agent_a(state: ConcurrentState) -> Dict[str, Any]:
        """Agent A: Fast and reliable"""
        start = time.time()
        time.sleep(0.5)  # Simulate work
        duration = time.time() - start
        
        results = state.get("agent_a_results", [])
        results.append(f"Agent A completed in {duration:.2f}s")
        
        return {
            "agent_a_results": results,
            "execution_times": {**state.get("execution_times", {}), "agent_a": duration}
        }
    
    def agent_b(state: ConcurrentState) -> Dict[str, Any]:
        """Agent B: Slow but reliable"""
        start = time.time()
        time.sleep(2.0)  # Simulate slow work
        duration = time.time() - start
        
        results = state.get("agent_b_results", [])
        results.append(f"Agent B completed in {duration:.2f}s")
        
        return {
            "agent_b_results": results,
            "execution_times": {**state.get("execution_times", {}), "agent_b": duration}
        }
    
    def agent_c(state: ConcurrentState) -> Dict[str, Any]:
        """Agent C: Unreliable, might fail"""
        start = time.time()
        
        # 30% chance of failure
        if random.random() < 0.3:
            raise Exception("Agent C failed randomly!")
        
        time.sleep(1.0)
        duration = time.time() - start
        
        results = state.get("agent_c_results", [])
        results.append(f"Agent C completed in {duration:.2f}s")
        
        return {
            "agent_c_results": results,
            "execution_times": {**state.get("execution_times", {}), "agent_c": duration}
        }
    
    # Build concurrent graph
    print("\n📊 Building concurrent workflow with 3 agents...")
    
    workflow = StateGraph(ConcurrentState)
    
    # Add all agents as separate nodes
    workflow.add_node("agent_a", agent_a)
    workflow.add_node("agent_b", agent_b)
    workflow.add_node("agent_c", agent_c)
    
    # Set all agents to start in parallel
    workflow.add_edge("__start__", "agent_a")
    workflow.add_edge("__start__", "agent_b")
    workflow.add_edge("__start__", "agent_c")
    
    # Add edges to END (they run in parallel)
    workflow.add_edge("agent_a", END)
    workflow.add_edge("agent_b", END)
    workflow.add_edge("agent_c", END)
    
    # Compile with concurrency configuration
    print("\n⚡ Running agents with max_concurrency...")
    
    try:
        app = workflow.compile()
        
        # Run multiple times to see different outcomes
        for i in range(3):
            print(f"\n--- Run {i+1} ---")
            
            initial_state = {
                "agent_a_results": [],
                "agent_b_results": [],
                "agent_c_results": [],
                "errors": [],
                "execution_times": {}
            }
            
            try:
                start_time = time.time()
                result = app.invoke(initial_state)
                total_time = time.time() - start_time
                
                print(f"Total execution time: {total_time:.2f}s")
                print(f"Agent A: {result.get('agent_a_results', ['failed'])[-1]}")
                print(f"Agent B: {result.get('agent_b_results', ['failed'])[-1]}")
                print(f"Agent C: {result.get('agent_c_results', ['failed'])[-1]}")
                
                # Check if agents ran in parallel
                max_agent_time = max(result.get('execution_times', {}).values(), default=0)
                print(f"Longest agent time: {max_agent_time:.2f}s")
                print(f"Parallel efficiency: {max_agent_time/total_time:.1%}")
                
            except Exception as e:
                print(f"Run failed: {e}")
                
    except Exception as e:
        print(f"Error building graph: {e}")
    
    # Demonstrate error isolation
    print("\n" + "-" * 70)
    print("🔒 ERROR ISOLATION DEMONSTRATION")
    print("-" * 70)
    
    # Create a graph where one agent fails
    class ErrorState(TypedDict):
        outputs: Dict[str, str]
        errors: Dict[str, str]
    
    def reliable_agent(state: ErrorState) -> Dict[str, Any]:
        outputs = state.get("outputs", {})
        outputs["reliable"] = "Completed successfully"
        return {"outputs": outputs}
    
    def failing_agent(state: ErrorState) -> Dict[str, Any]:
        raise Exception("This agent is designed to fail!")
    
    def independent_agent(state: ErrorState) -> Dict[str, Any]:
        outputs = state.get("outputs", {})
        outputs["independent"] = "Ran despite other failures"
        return {"outputs": outputs}
    
    # Build error-isolated graph
    error_workflow = StateGraph(ErrorState)
    error_workflow.add_node("reliable", reliable_agent)
    error_workflow.add_node("failing", failing_agent)
    error_workflow.add_node("independent", independent_agent)
    
    # Try to run them in sequence (failing agent will break chain)
    error_workflow.add_edge("__start__", "reliable")
    error_workflow.add_edge("reliable", "failing")
    error_workflow.add_edge("failing", "independent")  # This won't be reached
    
    try:
        error_app = error_workflow.compile()
        result = error_app.invoke({"outputs": {}, "errors": {}})
        print("Unexpected success!")
    except Exception as e:
        print(f"✅ Correctly caught error: {type(e).__name__}")
        print("The failing agent stopped the entire chain.")
    
    # Now demonstrate parallel execution with error handling
    print("\n🔄 PARALLEL EXECUTION WITH ERROR HANDLING")
    print("-" * 70)
    
    parallel_workflow = StateGraph(ErrorState)
    parallel_workflow.add_node("reliable", reliable_agent)
    parallel_workflow.add_node("failing", failing_agent)
    parallel_workflow.add_node("independent", independent_agent)
    
    # Set all as entry points (they run in parallel)
    parallel_workflow.add_edge("__start__", "reliable")
    parallel_workflow.add_edge("__start__", "failing")
    parallel_workflow.add_edge("__start__", "independent")
    
    # All go directly to END
    parallel_workflow.add_edge("reliable", END)
    parallel_workflow.add_edge("failing", END)
    parallel_workflow.add_edge("independent", END)
    
    try:
        parallel_app = parallel_workflow.compile()
        result = parallel_app.invoke({"outputs": {}, "errors": {}})
        print("✅ Parallel execution completed!")
        print(f"Successful outputs: {result.get('outputs', {})}")
        # Note: The failing agent's error might be lost or handled differently
        # depending on LangGraph's error handling configuration
    except Exception as e:
        print(f"Parallel execution failed: {e}")

# ============== 8. Advanced: Configurable Recursion Limit ==============
class SelfCorrectingAgent:
    """Advanced agent with configurable recursion limits and error handling."""
    
    def __init__(self, max_attempts: int = 5, use_checkpointer: bool = False):
        self.max_attempts = max_attempts
        self.use_checkpointer = use_checkpointer
        self.app = self._build_agent()
    
    def _build_agent(self):
        """Build agent with proper recursion handling."""
        
        workflow = StateGraph(SelfCorrectingState)
        
        # Enhanced searcher with attempt tracking
        def enhanced_searcher(state: SelfCorrectingState) -> Dict[str, Any]:
            attempt = state["attempt_count"] + 1
            
            # Check recursion limit
            if attempt > self.max_attempts:
                raise RecursionError(f"Maximum attempts ({self.max_attempts}) exceeded")
            
            # Simulate search with attempt-based quality
            if attempt < 3:
                result = f"Attempt {attempt}: Low quality"
                quality = "low"
            else:
                result = f"Attempt {attempt}: High quality data found"
                quality = "high"
            
            # Log this attempt
            error_log = state.get("error_log", [])
            error_log.append(f"Attempt {attempt}: {quality}")
            
            return {
                "attempt_count": attempt,
                "last_search_quality": quality,
                "search_results": state.get("search_results", []) + [result],
                "error_log": error_log,
                "messages": state["messages"] + [
                    AIMessage(content=f"Search: {result}")
                ]
            }
        
        # Enhanced analyzer with concurrency awareness
        def enhanced_analyzer(state: SelfCorrectingState) -> Dict[str, Any]:
            quality = state.get("last_search_quality", "low")
            
            if quality == "low":
                # Provide context-aware hint
                hints = [
                    "Try broadening your search terms",
                    "Consider alternative data sources",
                    "Check for recent updates or news",
                    "Verify the accuracy of your query",
                    "Consult expert databases"
                ]
                
                hint_idx = min(state["attempt_count"], len(hints) - 1)
                hint = hints[hint_idx]
                
                return {
                    "retry_hint": hint,
                    "messages": state["messages"] + [
                        SystemMessage(content=f"Analysis needed: {hint}")
                    ]
                }
            else:
                # Success - generate final answer
                return {
                    "messages": state["messages"] + [
                        AIMessage(content="✅ Search successful! Here are the comprehensive findings...")
                    ]
                }
        
        # Enhanced router with better error handling
        def enhanced_router(state: SelfCorrectingState) -> str:
            quality = state.get("last_search_quality", "low")
            
            if quality == "low":
                return "searcher"
            else:
                return "__end__"
        
        # Build the graph
        workflow.add_node("searcher", enhanced_searcher)
        workflow.add_node("analyzer", enhanced_analyzer)
        
        workflow.add_edge("__start__", "searcher")
        workflow.add_edge("searcher", "analyzer")
        
        workflow.add_conditional_edges(
            "analyzer",
            enhanced_router,
            {"searcher": "searcher", "__end__": END}
        )
        
        # Compile with optional memory for state persistence
        if self.use_checkpointer:
            memory = MemorySaver()
            return workflow.compile(checkpointer=memory)
        else:
            return workflow.compile()
    
    def search(self, query: str, thread_id: str = None, max_retries: int = None):
        """Execute search with retry logic."""
        
        if max_retries is not None:
            # Create a new instance with custom retry limit
            agent = SelfCorrectingAgent(max_attempts=max_retries, use_checkpointer=self.use_checkpointer)
            app = agent.app
        else:
            app = self.app
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "attempt_count": 0,
            "search_query": query,
            "search_results": [],
            "last_search_quality": "low",
            "retry_hint": "",
            "recursion_depth": 0,
            "error_log": []
        }
        
        # Generate thread_id if not provided
        if thread_id is None:
            thread_id = f"search_{uuid.uuid4().hex[:8]}"
        
        # Prepare config
        if self.use_checkpointer:
            config = {"configurable": {"thread_id": thread_id}}
        else:
            config = {}
        
        try:
            result = app.invoke(initial_state, config)
            return {
                "success": True,
                "attempts": result["attempt_count"],
                "results": result["search_results"],
                "final_message": result["messages"][-1].content,
                "error_log": result.get("error_log", []),
                "thread_id": thread_id
            }
        except RecursionError as e:
            return {
                "success": False,
                "error": str(e),
                "attempts": self.max_attempts,
                "results": [],
                "error_log": [f"RecursionError: {e}"],
                "thread_id": thread_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "attempts": 0,
                "results": [],
                "error_log": [f"Unexpected error: {e}"],
                "thread_id": thread_id
            }

# ============== 9. Main Execution ==============
if __name__ == "__main__":
    print("=" * 70)
    print("🔄 SELF-CORRECTING AGENT WITH RECURSION LIMITS")
    print("=" * 70)
    
    print("\nOptions:")
    print("1. Test self-correction mechanism")
    print("2. Study max_concurrency in LangGraph")
    print("3. Run advanced agent with configurable limits")
    print("4. Run comprehensive demo")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        test_self_correction()
    elif choice == "2":
        study_max_concurrency()
    elif choice == "3":
        print("\n" + "="*70)
        print("🤖 ADVANCED SELF-CORRECTING AGENT")
        print("="*70)
        
        # Create agent with different limits
        agent_3 = SelfCorrectingAgent(max_attempts=3, use_checkpointer=False)
        agent_5 = SelfCorrectingAgent(max_attempts=5, use_checkpointer=False)
        agent_10 = SelfCorrectingAgent(max_attempts=10, use_checkpointer=False)
        
        queries = [
            "Find research papers about AI ethics",
            "Get current stock market trends",
            "Learn about quantum computing applications"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 50)
            
            for agent_name, agent in [("3 attempts", agent_3), 
                                      ("5 attempts", agent_5),
                                      ("10 attempts", agent_10)]:
                print(f"\nTesting with {agent_name}:")
                result = agent.search(query)
                
                if result["success"]:
                    print(f"  ✅ Success after {result['attempts']} attempts")
                    print(f"  Final: {result['final_message'][:80]}...")
                else:
                    print(f"  ❌ Failed: {result['error']}")
                    print(f"  Attempts: {result['attempts']}")
    
    elif choice == "4":
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE DEMO")
        print("="*70)
        
        # Part 1: Self-correction
        print("\n📊 PART 1: Testing self-correction...")
        test_self_correction()
        
        # Part 2: Concurrency study
        print("\n\n📊 PART 2: Studying concurrency...")
        study_max_concurrency()
        
        # Part 3: Advanced agent with checkpointer
        print("\n\n📊 PART 3: Advanced agent with checkpointer...")
        advanced_agent = SelfCorrectingAgent(max_attempts=4, use_checkpointer=True)
        
        results = []
        for i in range(3):
            print(f"\n--- Search {i+1} ---")
            result = advanced_agent.search(f"Test query {i+1}", thread_id=f"thread_{i}")
            results.append(result)
            
            if result["success"]:
                print(f"✅ Success in {result['attempts']} attempts")
                print(f"  Thread ID: {result['thread_id']}")
            else:
                print(f"❌ Failed: {result['error']}")
                print(f"  Thread ID: {result['thread_id']}")
        
        print("\n" + "="*70)
        print("📈 SUMMARY")
        print("="*70)
        successful = sum(1 for r in results if r["success"])
        total_attempts = sum(r["attempts"] for r in results)
        print(f"Successful searches: {successful}/{len(results)}")
        print(f"Total attempts across all searches: {total_attempts}")
        print("="*70)
    
    else:
        print("Running default test...")
        test_self_correction()