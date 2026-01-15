from typing import TypedDict, List, Annotated, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
import json
import re

# ============== 1. Define the State Schema ==============
class RouterState(TypedDict):
    """State for the Smart Router agent."""
    messages: Annotated[List[BaseMessage], add_messages]  # Conversation history
    decision: str  # Oracle's decision (WEATHER, STOCK, TALK)
    tool_result: Optional[str]  # Result from the worker nodes
    user_query: str  # Original user query for reference

# ============== 2. Define the Tools ==============
def get_weather(location: str = "San Francisco") -> str:
    """Fake weather tool - returns weather information."""
    weather_data = {
        "San Francisco": "Sunny, 72°F, Wind: 5 mph",
        "New York": "Cloudy, 65°F, Wind: 10 mph",
        "London": "Rainy, 55°F, Wind: 15 mph",
        "Tokyo": "Clear, 75°F, Wind: 7 mph",
        "Sydney": "Windy, 80°F, Wind: 20 mph"
    }
    
    # Try to extract location from query if not provided
    if not location:
        location = "San Francisco"
    
    location = location.title()
    
    if location in weather_data:
        return f"Weather in {location}: {weather_data[location]}"
    else:
        # Generate fake weather for unknown location
        return f"Weather in {location}: Partly cloudy, 70°F, Wind: 8 mph"

def get_stock_price(ticker: str = "AAPL") -> str:
    """Fake stock price tool - returns stock information."""
    stock_data = {
        "AAPL": {"price": 175.25, "change": "+2.50", "change_percent": "+1.45%"},
        "GOOGL": {"price": 142.80, "change": "-0.75", "change_percent": "-0.52%"},
        "MSFT": {"price": 330.45, "change": "+1.20", "change_percent": "+0.36%"},
        "TSLA": {"price": 245.60, "change": "-3.25", "change_percent": "-1.31%"},
        "AMZN": {"price": 155.75, "change": "+0.95", "change_percent": "+0.61%"}
    }
    
    # Clean and standardize ticker
    ticker = ticker.upper().strip()
    
    if ticker in stock_data:
        data = stock_data[ticker]
        return f"{ticker}: ${data['price']} ({data['change']}, {data['change_percent']})"
    else:
        # Generate fake data for unknown ticker
        fake_price = round(100 + (hash(ticker) % 100), 2)
        fake_change = round((hash(ticker) % 20 - 10) / 10, 2)
        fake_pct = round(fake_change / fake_price * 100, 2)
        sign = "+" if fake_change >= 0 else ""
        return f"{ticker}: ${fake_price} ({sign}{fake_change}, {sign}{fake_pct}%)"

# ============== 3. Define the Oracle Node ==============
def oracle_node(state: RouterState) -> RouterState:
    """
    Oracle Node: LLM that analyzes user query and decides what to do.
    Returns a decision (WEATHER, STOCK, TALK) and sometimes extracts parameters.
    """
    
    # Get the latest user message
    user_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    if not user_query:
        return {"decision": "TALK", "user_query": ""}
    
    # Initialize LLM for decision making
    llm = ChatOllama(
        model="llama3",
        temperature=0.1,  # Low temperature for consistent decisions
        num_predict=128,
    )
    
    # Create decision prompt
    decision_prompt = f"""
    Analyze the user's query and decide which action to take:
    
    User Query: "{user_query}"
    
    Available Actions:
    1. WEATHER - If the user asks about weather, temperature, forecast, climate, etc.
    2. STOCK - If the user asks about stock prices, shares, trading, ticker symbols, etc.
    3. TALK - If it's a general conversation, greeting, or doesn't fit the other categories
    
    Respond in EXACTLY this JSON format:
    {{
        "decision": "WEATHER" or "STOCK" or "TALK",
        "reason": "Brief explanation of your decision",
        "parameters": {{
            "location": "city name" (if WEATHER decision),
            "ticker": "stock symbol" (if STOCK decision)
        }}
    }}
    
    Only include parameters relevant to your decision. If TALK, parameters should be empty.
    """
    
    # Get decision from LLM
    try:
        response = llm.invoke([HumanMessage(content=decision_prompt)])
        
        # Parse JSON response
        response_text = response.content.strip()
        
        # Clean up response - extract JSON even if there's extra text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            decision_data = json.loads(json_match.group())
        else:
            # Fallback if no JSON found
            decision_data = {
                "decision": "TALK",
                "reason": "Could not parse LLM response",
                "parameters": {}
            }
        
        # Extract location/ticker from user query as fallback
        user_query_lower = user_query.lower()
        
        # Weather locations
        weather_locations = ["san francisco", "new york", "london", "tokyo", "sydney", 
                           "paris", "berlin", "mumbai", "beijing", "toronto"]
        
        # Stock tickers
        stock_tickers = ["aapl", "googl", "msft", "tsla", "amzn", "meta", "nflx", "nvda", "amd", "intc"]
        
        location = None
        ticker = None
        
        # Try to extract location from query
        for loc in weather_locations:
            if loc in user_query_lower:
                location = loc.title()
                break
        
        # Try to extract ticker from query
        words = user_query_lower.split()
        for word in words:
            if word.upper() in [t.upper() for t in stock_tickers]:
                ticker = word.upper()
                break
        
        # If location/ticker not in JSON but implied by decision, use extracted ones
        if decision_data["decision"] == "WEATHER" and not decision_data["parameters"].get("location"):
            if location:
                decision_data["parameters"]["location"] = location
            else:
                decision_data["parameters"]["location"] = "San Francisco"
        
        if decision_data["decision"] == "STOCK" and not decision_data["parameters"].get("ticker"):
            if ticker:
                decision_data["parameters"]["ticker"] = ticker
            else:
                decision_data["parameters"]["ticker"] = "AAPL"
        
        print("=" * 70)
        print("🔮 ORACLE NODE - Decision Making:")
        print("-" * 70)
        print(f"User Query: {user_query}")
        print(f"Decision: {decision_data['decision']}")
        print(f"Reason: {decision_data['reason']}")
        print(f"Parameters: {decision_data['parameters']}")
        print("=" * 70)
        
        # Store parameters in messages as a system message for worker nodes
        param_message = None
        if decision_data["decision"] == "WEATHER":
            param_message = f"location:{decision_data['parameters'].get('location', 'San Francisco')}"
        elif decision_data["decision"] == "STOCK":
            param_message = f"ticker:{decision_data['parameters'].get('ticker', 'AAPL')}"
        
        if param_message:
            state["messages"].append(AIMessage(content=param_message))
        
        return {
            "decision": decision_data["decision"],
            "user_query": user_query,
            "messages": state["messages"]  # Preserve messages
        }
        
    except Exception as e:
        print(f"Error in oracle_node: {e}")
        return {"decision": "TALK", "user_query": user_query}

# ============== 4. Define Worker Nodes ==============
def weather_worker_node(state: RouterState) -> RouterState:
    """Process weather requests."""
    
    # Extract location from the last AI message (where Oracle stored it)
    location = "San Francisco"  # Default
    
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content.startswith("location:"):
            location = msg.content.split(":")[1].strip()
            break
    
    print("=" * 70)
    print("🌤️  WEATHER WORKER NODE:")
    print("-" * 70)
    print(f"Processing weather request for: {location}")
    
    # Call the weather tool
    weather_result = get_weather(location)
    
    print(f"Weather Result: {weather_result}")
    print("=" * 70)
    
    # Add result to messages as a system message
    result_message = AIMessage(content=f"Weather Tool Result: {weather_result}")
    
    return {
        "tool_result": weather_result,
        "messages": state["messages"] + [result_message]
    }

def stock_worker_node(state: RouterState) -> RouterState:
    """Process stock price requests."""
    
    # Extract ticker from the last AI message
    ticker = "AAPL"  # Default
    
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content.startswith("ticker:"):
            ticker = msg.content.split(":")[1].strip()
            break
    
    print("=" * 70)
    print("📈 STOCK WORKER NODE:")
    print("-" * 70)
    print(f"Processing stock request for: {ticker}")
    
    # Call the stock tool
    stock_result = get_stock_price(ticker)
    
    print(f"Stock Result: {stock_result}")
    print("=" * 70)
    
    # Add result to messages
    result_message = AIMessage(content=f"Stock Tool Result: {stock_result}")
    
    return {
        "tool_result": stock_result,
        "messages": state["messages"] + [result_message]
    }

# ============== 5. Custom Routing Function ==============
def custom_router(state: RouterState) -> str:
    """
    Custom routing function that parses the LLM output manually.
    Determines which node to go to next based on the Oracle's decision.
    """
    
    decision = state.get("decision", "TALK")
    
    print("\n" + "=" * 70)
    print("🔄 CUSTOM ROUTER - Making Routing Decision:")
    print("-" * 70)
    print(f"Oracle Decision: {decision}")
    
    # Manual routing logic
    if decision == "WEATHER":
        print("Routing to: WEATHER_WORKER")
        print("=" * 70)
        return "weather_worker"
    elif decision == "STOCK":
        print("Routing to: STOCK_WORKER")
        print("=" * 70)
        return "stock_worker"
    else:  # TALK or anything else
        print("Routing to: END (direct conversation)")
        print("=" * 70)
        return "__end__"

# ============== 6. Build the Graph ==============
def build_smart_router():
    """Build the Smart Router graph with custom routing."""
    
    # Create the workflow
    workflow = StateGraph(RouterState)
    
    # Add nodes
    workflow.add_node("oracle", oracle_node)
    workflow.add_node("weather_worker", weather_worker_node)
    workflow.add_node("stock_worker", stock_worker_node)
    
    # Set the entry point
    workflow.set_entry_point("oracle")
    
    # Add conditional edges using our custom router
    # This is the key part - using our custom function instead of tools_condition
    workflow.add_conditional_edges(
        "oracle",
        custom_router,
        {
            "weather_worker": "weather_worker",
            "stock_worker": "stock_worker", 
            "__end__": END
        }
    )
    
    # Add edges from worker nodes back to END
    workflow.add_edge("weather_worker", END)
    workflow.add_edge("stock_worker", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# ============== 7. Response Formatter Node (Optional Enhancement) ==============
def response_formatter_node(state: RouterState) -> RouterState:
    """
    Optional: Format the final response nicely for the user.
    This shows how you could extend the graph further.
    """
    
    last_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_message = msg.content
            break
    
    # Format based on tool result
    tool_result = state.get("tool_result", "")
    user_query = state.get("user_query", "")
    
    if "Weather Tool Result:" in last_message:
        # Extract weather info
        weather_info = last_message.replace("Weather Tool Result: ", "")
        formatted = f"🌤️  Here's the weather information you requested:\n\n{weather_info}"
    elif "Stock Tool Result:" in last_message:
        # Extract stock info
        stock_info = last_message.replace("Stock Tool Result: ", "")
        formatted = f"📈 Here's the stock information you requested:\n\n{stock_info}"
    else:
        # General conversation - use the LLM's original response
        formatted = last_message
    
    # Add formatted response
    formatted_message = AIMessage(content=formatted)
    
    return {
        "messages": [formatted_message],
        "tool_result": tool_result
    }

# ============== 8. Enhanced Graph with Formatter ==============
def build_enhanced_smart_router():
    """Build enhanced graph with response formatting."""
    
    workflow = StateGraph(RouterState)
    
    # Add all nodes
    workflow.add_node("oracle", oracle_node)
    workflow.add_node("weather_worker", weather_worker_node)
    workflow.add_node("stock_worker", stock_worker_node)
    workflow.add_node("formatter", response_formatter_node)
    
    workflow.set_entry_point("oracle")
    
    # Custom routing from oracle
    workflow.add_conditional_edges(
        "oracle",
        custom_router,
        {
            "weather_worker": "weather_worker",
            "stock_worker": "stock_worker",
            "__end__": "formatter"  # Go to formatter for direct responses
        }
    )
    
    # Route from workers to formatter
    workflow.add_edge("weather_worker", "formatter")
    workflow.add_edge("stock_worker", "formatter")
    
    # Formatter goes to END
    workflow.add_edge("formatter", END)
    
    return workflow.compile()

# ============== 9. Test Function ==============
def test_smart_router():
    """Test the Smart Router with various queries."""
    
    print("🚀 SMART ROUTER AGENT - Custom Routing Demo")
    print("=" * 70)
    
    # Build the basic router
    router = build_smart_router()
    
    test_queries = [
        "What's the weather like in Tokyo?",
        "How is AAPL stock doing today?",
        "Tell me about the weather forecast",
        "What's the stock price of Tesla?",
        "Hello, how are you?",
        "Can you check Microsoft stock for me?",
        "Is it raining in London?",
        "I want to know about GOOGL shares",
        "What's up?",
        "Check weather for Paris and stock for Amazon"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: '{query}'")
        print('='*70)
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "decision": "",
            "tool_result": None,
            "user_query": query
        }
        
        # Invoke the router
        result = router.invoke(initial_state)
        
        # Display final response
        print("\n" + "-" * 70)
        print("💬 FINAL RESPONSE:")
        print("-" * 70)
        
        # Find the last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                # Skip parameter/tool result messages
                if not (msg.content.startswith("location:") or 
                        msg.content.startswith("ticker:") or
                        msg.content.startswith("Weather Tool Result:") or
                        msg.content.startswith("Stock Tool Result:")):
                    print(f"Assistant: {msg.content}")
                    break
        
        print("\n📊 FINAL STATE SUMMARY:")
        print(f"Decision: {result.get('decision', 'N/A')}")
        print(f"Tool Result: {result.get('tool_result', 'None')}")
        print(f"Total Messages: {len(result.get('messages', []))}")
        print("=" * 70)
        
        # Pause between tests
        if i < len(test_queries):
            input("\nPress Enter for next test...")

# ============== 10. Interactive Chat ==============
def interactive_chat():
    """Interactive chat with the Smart Router."""
    
    print("🤖 SMART ROUTER - Interactive Chat Mode")
    print("=" * 70)
    print("Ask about weather, stocks, or have a conversation!")
    print("Type 'quit' to exit, 'state' to see current state")
    print("=" * 70)
    
    # Build enhanced router with formatter
    router = build_enhanced_smart_router()
    
    # Start with empty state
    state = {
        "messages": [],
        "decision": "",
        "tool_result": None,
        "user_query": ""
    }
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'state':
            print("\n📊 CURRENT STATE:")
            print(f"Decision: {state.get('decision', 'None')}")
            print(f"Tool Result: {state.get('tool_result', 'None')}")
            print(f"Message Count: {len(state.get('messages', []))}")
            continue
        
        # Add user message
        state["messages"].append(HumanMessage(content=user_input))
        state["user_query"] = user_input
        
        # Invoke router
        result = router.invoke(state)
        
        # Update state
        state.update(result)
        
        # Display assistant's response
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and not (
                msg.content.startswith("location:") or
                msg.content.startswith("ticker:") or
                msg.content.startswith("Weather Tool Result:") or
                msg.content.startswith("Stock Tool Result:")
            ):
                print(f"\nAssistant: {msg.content}")
                break

# ============== 11. Visualization Helper ==============
def visualize_workflow():
    """Generate Mermaid diagram of the workflow."""
    
    mermaid_diagram = """
    ```mermaid
    graph TD
        A[User Input] --> B[Oracle Node]
        B --> C{Custom Router}
        
        C -->|WEATHER| D[Weather Worker]
        C -->|STOCK| E[Stock Worker]
        C -->|TALK| F[END]
        
        D --> F
        E --> F
        
        subgraph "Oracle Node Logic"
            B1[LLM analyzes query]
            B2[Extracts parameters]
            B3[Makes decision]
            B1 --> B2 --> B3
        end
        
        subgraph "Custom Router Logic"
            C1[Reads decision]
            C2[Parses manually]
            C3[Routes accordingly]
            C1 --> C2 --> C3
        end
        
        style B fill:#e1f5fe
        style C fill:#fff3e0
        style D fill:#e8f5e8
        style E fill:#fce4ec
    ```
    """
    
    print("📊 WORKFLOW VISUALIZATION")
    print("=" * 70)
    print(mermaid_diagram)
    print("\nKey Points:")
    print("1. Oracle Node uses LLM to analyze query and decide action")
    print("2. Custom Router manually parses the decision (no tools_condition)")
    print("3. Routing: WEATHER → Weather Worker, STOCK → Stock Worker, TALK → END")
    print("4. Worker nodes call tools and return results")
    print("=" * 70)

# ============== 12. Main Execution ==============
if __name__ == "__main__":
    print("=" * 70)
    print("SMART ROUTER AGENT WITH CUSTOM ROUTING")
    print("=" * 70)
    
    print("\nOptions:")
    print("1. Run automated tests")
    print("2. Interactive chat mode")
    print("3. Visualize workflow")
    print("4. See custom routing logic details")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        test_smart_router()
    elif choice == "2":
        interactive_chat()
    elif choice == "3":
        visualize_workflow()
    elif choice == "4":
        print("\n" + "="*70)
        print("CUSTOM ROUTING LOGIC EXPLAINED")
        print("="*70)
        print("\nThe key function is `custom_router(state):`")
        print("\nCode:")
        print("-" * 40)
        print('''
def custom_router(state: RouterState) -> str:
    """
    Custom routing function that parses the LLM output manually.
    Determines which node to go to next based on the Oracle's decision.
    """
    
    decision = state.get("decision", "TALK")
    
    # Manual routing logic
    if decision == "WEATHER":
        return "weather_worker"
    elif decision == "STOCK":
        return "stock_worker"
    else:  # TALK or anything else
        return "__end__"
        ''')
        print("-" * 40)
        print("\nKey Differences from tools_condition:")
        print("1. We manually parse `state['decision']` (LLM output)")
        print("2. We return destination node names as strings")
        print("3. We handle the mapping in add_conditional_edges()")
        print("4. We control the entire flow manually")
        print("="*70)
    else:
        print("Running automated tests by default...")
        test_smart_router()