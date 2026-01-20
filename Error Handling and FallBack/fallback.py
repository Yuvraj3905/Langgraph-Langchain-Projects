from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import random
import time

# ============== 1. Define the State Schema ==============
class HealingState(TypedDict):
    """State for the self-healing agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    attempts: int
    max_attempts: int
    last_error: str
    retry_count: int
    used_fallback: bool

# ============== 2. Define the Unstable API ==============
class UnstableAPI:
    """Tool that fails 50% of the time."""
    
    def __init__(self, failure_rate: float = 0.5):
        self.failure_rate = failure_rate
        self.call_history = []
    
    def call(self, query: str = "default query") -> str:
        """
        Simulates an API call that fails 50% of the time.
        
        Returns:
            str: Success message or raises ValueError
        """
        # Simulate network latency
        time.sleep(0.1)
        
        # 50% chance of failure
        if random.random() < self.failure_rate:
            error = ValueError("API Connection Timeout: Server not responding")
            self.call_history.append(("FAIL", str(error)))
            raise error
        else:
            response = f"API Success: Retrieved data for '{query}'"
            self.call_history.append(("SUCCESS", response))
            return response

# ============== 3. Action Node with Try-Except ==============
def action_node(state: HealingState) -> Dict[str, Any]:
    """
    Node that calls the unstable API.
    Uses try-except to catch errors and update state.
    """
    print("=" * 60)
    print("🎯 ACTION NODE: Calling Unstable API...")
    
    api = UnstableAPI(failure_rate=0.5)
    query = "Get user data"  # Default query
    
    # Get user query from messages if available
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
    
    try:
        # Try to call the API
        result = api.call(query)
        
        print(f"✅ SUCCESS: {result}")
        print(f"   Attempts so far: {state.get('attempts', 0) + 1}")
        
        # Success - update state
        return {
            "attempts": state.get("attempts", 0) + 1,
            "last_error": "",  # Clear any previous error
            "retry_count": 0,  # Reset retry count on success
            "messages": state["messages"] + [AIMessage(content=result)]
        }
        
    except ValueError as e:
        # API call failed
        error_msg = str(e)
        print(f"❌ FAILURE: {error_msg}")
        print(f"   Retry count: {state.get('retry_count', 0) + 1}")
        
        # Update state with error
        return {
            "attempts": state.get("attempts", 0) + 1,
            "last_error": error_msg,
            "retry_count": state.get("retry_count", 0) + 1,
            "messages": state["messages"] + [
                AIMessage(content=f"API call failed: {error_msg}")
            ]
        }

# ============== 4. Repairman Node (The "Agentic" Fallback) ==============
def repairman_node(state: HealingState) -> Dict[str, Any]:
    """
    Repairman node - the "Agentic" way to handle failures.
    Analyzes error and decides recovery strategy.
    """
    print("=" * 60)
    print("🔧 REPAIRMAN NODE: Analyzing failure...")
    
    error = state.get("last_error", "Unknown error")
    retry_count = state.get("retry_count", 0)
    
    # Analyze error type
    error_lower = error.lower()
    
    if "timeout" in error_lower:
        strategy = "Timeout detected. Will retry with exponential backoff."
        recovery_type = "retry_with_backoff"
    elif "connection" in error_lower:
        strategy = "Connection issue. Trying alternative endpoint."
        recovery_type = "try_alternative"
    else:
        strategy = "Unknown error. Attempting standard retry."
        recovery_type = "standard_retry"
    
    print(f"   Error: {error}")
    print(f"   Strategy: {strategy}")
    print(f"   Recovery Type: {recovery_type}")
    
    # Add repair message to state
    repair_message = f"🔧 Repairman: I encountered an error: '{error}'. {strategy} Let me try a different approach."
    
    return {
        "messages": state["messages"] + [
            SystemMessage(content=repair_message)
        ]
    }

# ============== 5. Graceful Failure Node ==============
def graceful_failure_node(state: HealingState) -> Dict[str, Any]:
    """
    Graceful Failure node - handles complete failure scenarios.
    Implements three recovery strategies.
    """
    print("=" * 60)
    print("🛡️  GRACEFUL FAILURE NODE: All retries exhausted")
    
    error = state.get("last_error", "Unknown error")
    attempts = state.get("attempts", 0)
    
    # Choose recovery strategy based on context
    strategies = [
        {
            "name": "Apologize and suggest alternative",
            "message": f"I'm sorry, after {attempts} attempts I couldn't connect to the API. Please try again later or contact support.",
            "type": "apology"
        },
        {
            "name": "Try cheaper Ollama model",
            "message": "API unavailable. Switching to local Ollama model (llama2) for response... [Simulated local response]",
            "type": "cheaper_model"
        },
        {
            "name": "Ask for human help",
            "message": "I'm having persistent API issues. Would you like me to: 1) Save this request for later, 2) Try a different query, or 3) Escalate to human support?",
            "type": "human_help"
        }
    ]
    
    # Select strategy based on error type and retry count
    if attempts >= 5:
        strategy = strategies[2]  # Human help for many failures
    elif "timeout" in error.lower():
        strategy = strategies[1]  # Cheaper model for timeouts
    else:
        strategy = strategies[0]  # Apology for other errors
    
    print(f"   Selected Strategy: {strategy['name']}")
    print(f"   Total Attempts: {attempts}")
    
    return {
        "used_fallback": True,
        "messages": state["messages"] + [
            AIMessage(content=strategy["message"])
        ]
    }

# ============== 6. Custom Router ==============
def healing_router(state: HealingState) -> str:
    """
    Custom router that implements circuit breaker logic.
    Determines next node based on error state and retry count.
    """
    last_error = state.get("last_error", "")
    retry_count = state.get("retry_count", 0)
    max_attempts = state.get("max_attempts", 3)
    
    print("\n" + "=" * 60)
    print("🔄 HEALING ROUTER: Making decision")
    print(f"   Has error: {'Yes' if last_error else 'No'}")
    print(f"   Retry count: {retry_count}/{max_attempts}")
    
    if last_error:  # There was an error
        if retry_count < max_attempts:
            print("   Decision: → REPAIRMAN (retry available)")
            return "repairman"
        else:
            print("   Decision: → GRACEFUL_FAILURE (max retries reached)")
            return "graceful_failure"
    else:  # No error
        print("   Decision: → END (success)")
        return "__end__"

# ============== 7. Build the Self-Healing Graph ==============
def build_self_healing_graph(max_attempts: int = 3):
    """
    Builds the self-healing agent graph with proper error handling.
    
    Args:
        max_attempts: Maximum number of retry attempts before graceful failure
    """
    
    workflow = StateGraph(HealingState)
    
    # Add all nodes
    workflow.add_node("action", action_node)
    workflow.add_node("repairman", repairman_node)
    workflow.add_node("graceful_failure", graceful_failure_node)
    
    # Set entry point
    workflow.set_entry_point("action")
    
    # Add conditional routing from action node
    workflow.add_conditional_edges(
        "action",
        healing_router,
        {
            "repairman": "repairman",
            "graceful_failure": "graceful_failure",
            "__end__": END
        }
    )
    
    # Add edges from other nodes
    workflow.add_edge("repairman", "action")  # Retry after repair
    workflow.add_edge("graceful_failure", END)  # Graceful failure ends
    
    # Compile the graph
    # Note: LangGraph's Pregel engine has built-in recursion protection
    app = workflow.compile()
    
    return app

# ============== 8. Test the Agent ==============
def test_self_healing_agent():
    """Test the self-healing agent with multiple runs."""
    
    print("\n" + "=" * 60)
    print("🧪 TESTING SELF-HEALING AGENT")
    print("=" * 60)
    
    # Build agent with 3 retry attempts
    agent = build_self_healing_graph(max_attempts=3)
    
    # Run multiple tests
    test_results = []
    
    for i in range(10):
        print(f"\n{'='*60}")
        print(f"TEST RUN {i+1}/10")
        print('='*60)
        
        initial_state = {
            "messages": [HumanMessage(content="Get user profile data")],
            "attempts": 0,
            "max_attempts": 3,
            "last_error": "",
            "retry_count": 0,
            "used_fallback": False
        }
        
        try:
            result = agent.invoke(initial_state)
            
            # Analyze result
            success = result.get("last_error", "") == ""
            attempts = result.get("attempts", 0)
            used_fallback = result.get("used_fallback", False)
            
            if success:
                status = "✅ SUCCESS"
                details = f"after {attempts} attempt(s)"
            elif used_fallback:
                status = "🛡️  GRACEFUL FAILURE"
                details = f"used fallback after {attempts} attempts"
            else:
                status = "❌ UNKNOWN"
                details = ""
            
            print(f"Result: {status} {details}")
            
            test_results.append({
                "run": i + 1,
                "success": success,
                "attempts": attempts,
                "used_fallback": used_fallback,
                "final_message": result["messages"][-1].content[:50] + "..." 
                if result["messages"] else "No message"
            })
            
        except Exception as e:
            print(f"❌ EXCEPTION: {type(e).__name__}: {e}")
            test_results.append({
                "run": i + 1,
                "error": str(e),
                "success": False
            })
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY (10 runs)")
    print("=" * 60)
    
    successful = sum(1 for r in test_results if r.get("success", False))
    fallbacks = sum(1 for r in test_results if r.get("used_fallback", False))
    errors = sum(1 for r in test_results if "error" in r)
    
    total_attempts = sum(r.get("attempts", 0) for r in test_results)
    avg_attempts = total_attempts / len(test_results) if test_results else 0
    
    print(f"Successful runs: {successful}/10 ({successful/10*100:.0f}%)")
    print(f"Graceful failures: {fallbacks}/10 ({fallbacks/10*100:.0f}%)")
    print(f"Exceptions: {errors}/10 ({errors/10*100:.0f}%)")
    print(f"Average attempts per run: {avg_attempts:.1f}")
    print(f"Total API calls simulated: {total_attempts}")
    
    # Show a sample successful conversation
    print("\n" + "=" * 60)
    print("💬 SAMPLE SUCCESSFUL CONVERSATION")
    print("=" * 60)
    
    sample_state = {
        "messages": [HumanMessage(content="Get weather data")],
        "attempts": 0,
        "max_attempts": 3,
        "last_error": "",
        "retry_count": 0,
        "used_fallback": False
    }
    
    sample_result = agent.invoke(sample_state)
    
    print("\nConversation:")
    for msg in sample_result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"👤 Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"🤖 AI: {msg.content}")
        elif isinstance(msg, SystemMessage):
            print(f"🔧 System: {msg.content}")

# ============== 9. DEEP DIVE: GraphRecursionError Study ==============
def study_graph_recursion_error():
    """
    Study why GraphRecursionError is your best friend.
    """
    
    print("\n" + "=" * 60)
    print("🧠 DEEP DIVE: GraphRecursionError")
    print("=" * 60)
    
    print("\n📚 WHAT IS GraphRecursionError?")
    print("-" * 40)
    print("""
    GraphRecursionError is LangGraph's built-in safety mechanism
    that prevents infinite loops in graph execution.
    
    It's triggered when:
    1. A graph has a cycle (A → B → A → B ...)
    2. The cycle repeats beyond recursion_limit
    3. Default limit is typically 25-50 cycles
    
    Without this protection, a bug could cause:
    • Infinite API calls
    • Massive bills
    • Server crashes
    • Denial of service
    """)
    
    print("\n⚡ DEMONSTRATION: The Danger Without Limits")
    print("-" * 40)
    
    # Create a dangerous infinite loop graph
    from langgraph.graph import StateGraph
    
    class LoopState(TypedDict):
        count: int
    
    def infinite_node(state: LoopState):
        return {"count": state.get("count", 0) + 1}
    
    dangerous_graph = StateGraph(LoopState)
    dangerous_graph.add_node("infinite", infinite_node)
    dangerous_graph.add_edge("infinite", "infinite")  # Self-loop!
    dangerous_graph.set_entry_point("infinite")
    
    try:
        dangerous_app = dangerous_graph.compile()
        print("Created infinite loop graph...")
        print("Without GraphRecursionError, this would run forever!")
        print("LangGraph will stop it automatically after recursion_limit.")
    except Exception as e:
        print(f"✅ LangGraph protects us: {type(e).__name__}")
    
    print("\n💰 FINANCIAL IMPACT EXAMPLE")
    print("-" * 40)
    print("""
    Scenario: Paid API @ $0.01 per call
    
    BUGGY AGENT (no recursion limit):
    • Failing API → Retry immediately
    • 50% failure rate
    • Infinite loop
    
    Cost calculation:
    • 10 calls/second = $0.10/second
    • $6/minute = $360/hour
    • $8,640/day = bankruptcy! 💸
    
    SMART AGENT (with GraphRecursionError):
    • LangGraph detects infinite loop
    • Stops after recursion_limit (e.g., 25)
    • Max cost: $0.25 ✅
    • System remains stable
    """)
    
    print("\n🛡️ HOW TO USE IT EFFECTIVELY")
    print("-" * 40)
    print("""
    1. Rely on defaults:
       • LangGraph has sane defaults
       • Usually 25-50 recursion limit
    
    2. Set custom limits:
       app = workflow.compile(recursion_limit=10)
    
    3. Combine with your own circuit breaker:
       • Track retry_count in state
       • Router checks retry_count < max_retries
       • Double protection!
    
    4. Monitor in production:
       • Log when recursion_limit approached
       • Alert on potential infinite loops
       • Review graphs for accidental cycles
    """)
    
    print("\n🎯 KEY TAKEAWAY")
    print("-" * 40)
    print("""
    GraphRecursionError is NOT a bug - it's a FEATURE!
    
    It's your automated:
    • Financial protector
    • System stabilizer  
    • Infinite loop preventer
    • Production safety net
    
    Always build self-healing loops WITH recursion limits.
    Your future self (and wallet) will thank you! 🎯
    """)

# ============== 10. Interactive Demo ==============
def interactive_demo():
    """Interactive demo of the self-healing agent."""
    
    print("\n" + "=" * 60)
    print("🤖 INTERACTIVE SELF-HEALING AGENT")
    print("=" * 60)
    
    print("\nBuilding agent with:")
    print("• 50% API failure rate")
    print("• 3 retry attempts maximum")
    print("• Automatic error recovery")
    print("• Graceful failure strategies")
    
    agent = build_self_healing_graph(max_attempts=3)
    
    while True:
        print("\n" + "=" * 60)
        print("Options:")
        print("1. Test with a query")
        print("2. Run 10 automated tests")
        print("3. Learn about GraphRecursionError")
        print("4. Exit")
        
        choice = input("\nYour choice (1-4): ").strip()
        
        if choice == "1":
            query = input("Enter your query: ").strip()
            if not query:
                query = "Get user data"
            
            print(f"\n🧪 Testing with query: '{query}'")
            print("-" * 40)
            
            state = {
                "messages": [HumanMessage(content=query)],
                "attempts": 0,
                "max_attempts": 3,
                "last_error": "",
                "retry_count": 0,
                "used_fallback": False
            }
            
            result = agent.invoke(state)
            
            print("\n📋 RESULTS:")
            print("-" * 40)
            for msg in result["messages"]:
                if isinstance(msg, HumanMessage):
                    print(f"👤 You: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"🤖 AI: {msg.content[:80]}...")
                elif isinstance(msg, SystemMessage):
                    print(f"🔧 System: {msg.content[:80]}...")
            
            print(f"\n📊 Stats: {result.get('attempts', 0)} attempts, " 
                  f"Retries: {result.get('retry_count', 0)}, "
                  f"Fallback: {result.get('used_fallback', False)}")
            
        elif choice == "2":
            print("\nRunning 10 automated tests...")
            test_self_healing_agent()
            
        elif choice == "3":
            study_graph_recursion_error()
            
        elif choice == "4":
            print("\n👋 Goodbye! Remember: Always use circuit breakers!")
            break
            
        else:
            print("Invalid choice. Please enter 1-4.")

# ============== 11. Main Execution ==============
if __name__ == "__main__":
    print("=" * 60)
    print("🔄 SELF-HEALING API AGENT")
    print("=" * 60)
    print("\nDemonstrating:")
    print("• Try-Except pattern with agentic fallbacks")
    print("• Automatic error recovery")
    print("• GraphRecursionError as circuit breaker")
    
    print("\nChoose mode:")
    print("1. Run tests")
    print("2. Interactive demo")
    print("3. Deep dive on GraphRecursionError")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_self_healing_agent()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        study_graph_recursion_error()
    else:
        print("Running tests by default...")
        test_self_healing_agent()