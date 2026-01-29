from typing import TypedDict, List, Annotated, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import time
from datetime import datetime

# ============== 1. Define the State Schema ==============
class TransactionState(TypedDict):
    """State for transaction approval system."""
    amount: int
    action_status: str
    transaction_type: str
    recipient: str
    approval_required: bool
    approval_granted: Optional[bool]
    manager_notes: str
    execution_history: List[str]
    timestamp: str

# ============== 2. Define the Nodes ==============

# Node 1: Check Amount
def node_check(state: TransactionState) -> dict:
    """Check if amount requires manager approval."""
    
    print("\n" + "="*60)
    print("🔍 NODE_CHECK: Analyzing Transaction")
    print("="*60)
    
    amount = state["amount"]
    transaction_type = state["transaction_type"]
    recipient = state["recipient"]
    
    print(f"Transaction Details:")
    print(f"  Type: {transaction_type}")
    print(f"  Amount: ${amount}")
    print(f"  Recipient: {recipient}")
    
    # Determine if approval is needed
    approval_needed = amount > 100
    
    if approval_needed:
        print(f"\n⚠️  MANAGER APPROVAL REQUIRED!")
        print(f"  Reason: Amount (${amount}) exceeds $100 threshold")
        action_status = "PENDING_APPROVAL"
    else:
        print(f"\n✅ AUTO-APPROVED")
        print(f"  Reason: Amount (${amount}) is within $100 limit")
        action_status = "APPROVED"
    
    # Update history
    history = state.get("execution_history", [])
    history.append(f"Checked amount ${amount}: {action_status}")
    
    print("="*60)
    
    return {
        "approval_required": approval_needed,
        "action_status": action_status,
        "execution_history": history,
        "timestamp": datetime.now().isoformat()
    }

# Node 2: Execute Transaction
def node_execute(state: TransactionState) -> dict:
    """Execute the transaction."""
    
    print("\n" + "="*60)
    print("⚡ NODE_EXECUTE: Processing Transaction")
    print("="*60)
    
    amount = state["amount"]
    transaction_type = state["transaction_type"]
    recipient = state["recipient"]
    approval_granted = state.get("approval_granted", True)
    
    if not approval_granted:
        print("❌ TRANSACTION REJECTED!")
        print(f"  Manager denied approval for ${amount} to {recipient}")
        action_status = "REJECTED"
        message = f"Transaction of ${amount} to {recipient} was REJECTED by manager."
    else:
        print("✅ TRANSACTION SUCCESSFUL!")
        print(f"  Processed ${amount} {transaction_type} to {recipient}")
        action_status = "COMPLETED"
        message = f"Transaction Successful: ${amount} sent to {recipient}."
        
        # Simulate processing
        time.sleep(0.5)
        print(f"  Processing... Done!")
    
    # Update history
    history = state.get("execution_history", [])
    history.append(f"Executed: {action_status}")
    
    print("="*60)
    
    return {
        "action_status": action_status,
        "execution_history": history,
        "timestamp": datetime.now().isoformat(),
        "messages": [message] if 'messages' not in state else state['messages'] + [message]
    }

# Node 3: Request Approval (for manual intervention)
def node_request_approval(state: TransactionState) -> dict:
    """Simulate requesting manager approval."""
    
    print("\n" + "="*60)
    print("📋 NODE_REQUEST_APPROVAL: Manager Review Needed")
    print("="*60)
    
    amount = state["amount"]
    transaction_type = state["transaction_type"]
    recipient = state["recipient"]
    
    print(f"MANAGER APPROVAL REQUEST:")
    print(f"  Type: {transaction_type}")
    print(f"  Amount: ${amount}")
    print(f"  Recipient: {recipient}")
    print(f"\n⚠️  ACTION REQUIRED: This transaction needs manual approval!")
    print(f"\nTo approve, run: approve_transaction('{state.get('thread_id', 'default')}')")
    print(f"To reject, run: reject_transaction('{state.get('thread_id', 'default')}')")
    print("="*60)
    
    # Update history
    history = state.get("execution_history", [])
    history.append(f"Awaiting manager approval for ${amount}")
    
    return {
        "action_status": "AWAITING_MANUAL_APPROVAL",
        "execution_history": history,
        "timestamp": datetime.now().isoformat()
    }

# ============== 3. Build the Graph with Interrupts ==============
def build_transaction_agent(use_interrupt: bool = True):
    """Build transaction agent with optional interrupt."""
    
    workflow = StateGraph(TransactionState)
    
    # Add nodes
    workflow.add_node("check", node_check)
    workflow.add_node("execute", node_execute)
    workflow.add_node("auto_execute", node_execute)
    workflow.add_node("request_approval", node_request_approval)
    
    # Set entry point
    workflow.set_entry_point("check")
    
    # Add conditional edges
    def approval_router(state: TransactionState) -> str:
        """Route based on approval requirement."""
        
        approval_required = state.get("approval_required", False)
        approval_granted = state.get("approval_granted", None)
        
        print("\n" + "="*60)
        print("🔄 APPROVAL ROUTER: Making Decision")
        print("="*60)
        print(f"Approval required: {approval_required}")
        print(f"Approval granted: {approval_granted}")
        
        if not approval_required:
            print("Decision: → AUTO_EXECUTE (no approval needed)")
            return "auto_execute"
        elif approval_granted is True:
            print("Decision: → EXECUTE (manager approved)")
            return "execute"
        elif approval_granted is False:
            print("Decision: → END (manager rejected)")
            return "__end__"
        else:
            print("Decision: → REQUEST_APPROVAL (needs manager)")
            return "request_approval"
    
    # Add routing
    workflow.add_conditional_edges(
        "check",
        approval_router,
        {
            "execute": "execute",
            "auto_execute": "auto_execute",
            "request_approval": "request_approval",
            "__end__": END
        }
    )
    
    # Add edges
    workflow.add_edge("request_approval", "execute")  # Now goes to execute (where it will interrupt)
    workflow.add_edge("execute", END)
    workflow.add_edge("auto_execute", END)
    
    # Create memory saver
    memory = MemorySaver()
    
    # Compile with or without interrupt
    if use_interrupt:
        print("\n🔧 COMPILING WITH INTERRUPT_BEFORE ON 'execute'")
        app = workflow.compile(
            checkpointer=memory,
            interrupt_before=["execute"]  # MAGIC HAPPENS HERE!
        )
    else:
        print("\n🔧 COMPILING WITHOUT INTERRUPTS")
        app = workflow.compile(checkpointer=memory)
    
    return app

# ============== 4. Helper Functions for Manual Control ==============
class TransactionManager:
    """Manager class to handle transaction approvals."""
    
    def __init__(self, agent):
        self.agent = agent
        self.active_threads = {}
    
    def create_transaction(self, amount: int, transaction_type: str, 
                          recipient: str, thread_id: str = None) -> dict:
        """Create a new transaction request."""
        
        if not thread_id:
            thread_id = f"tx_{int(time.time())}_{amount}"
        
        initial_state = {
            "amount": amount,
            "action_status": "INITIALIZED",
            "transaction_type": transaction_type,
            "recipient": recipient,
            "approval_required": False,
            "approval_granted": None,
            "manager_notes": "",
            "execution_history": [],
            "timestamp": datetime.now().isoformat()
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n💸 CREATING TRANSACTION #{thread_id}")
        print(f"  Amount: ${amount}")
        print(f"  Type: {transaction_type}")
        print(f"  To: {recipient}")
        
        # Store thread info
        self.active_threads[thread_id] = {
            "state": initial_state,
            "config": config,
            "created": datetime.now().isoformat()
        }
        
        # Invoke agent
        result = self.agent.invoke(initial_state, config)
        
        # Update stored state
        self.active_threads[thread_id]["state"] = result
        
        return {"thread_id": thread_id, "result": result}
    
    def check_status(self, thread_id: str) -> dict:
        """Check the current status of a transaction."""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.agent.get_state(config)
            
            if not state or not state.values:
                return {"error": f"No transaction found with ID: {thread_id}"}
            
            print(f"\n📊 TRANSACTION STATUS: #{thread_id}")
            print(f"  Amount: ${state.values.get('amount', 0)}")
            print(f"  Status: {state.values.get('action_status', 'UNKNOWN')}")
            print(f"  Approval Required: {state.values.get('approval_required', False)}")
            print(f"  Approval Granted: {state.values.get('approval_granted', 'Not Set')}")
            
            if state.next:
                print(f"  ⚠️  WAITING AT NODE: {state.next}")
                print(f"  Next action required!")
            
            # Show history
            history = state.values.get("execution_history", [])
            if history:
                print(f"\n  Execution History:")
                for entry in history[-5:]:  # Last 5 entries
                    print(f"    • {entry}")
            
            return {
                "thread_id": thread_id,
                "state": state.values,
                "next": state.next,
                "is_waiting": bool(state.next)
            }
            
        except Exception as e:
            return {"error": f"Error checking status: {e}"}
    
    def approve_transaction(self, thread_id: str, notes: str = "") -> dict:
        """Approve a pending transaction."""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Get current state
            state = self.agent.get_state(config)
            
            if not state or not state.values:
                return {"error": f"No transaction found with ID: {thread_id}"}
            
            if not state.next:
                return {"error": f"Transaction #{thread_id} is not waiting for approval"}
            
            print(f"\n✅ MANAGER APPROVAL GRANTED: #{thread_id}")
            print(f"  Amount: ${state.values.get('amount', 0)}")
            print(f"  Notes: {notes}")
            
            # Update state with approval
            updated_state = dict(state.values)
            updated_state["approval_granted"] = True
            updated_state["manager_notes"] = notes
            updated_state["action_status"] = "APPROVED"
            
            # Resume execution
            result = self.agent.invoke(updated_state, config)
            
            print(f"  Result: {result.get('action_status', 'Unknown')}")
            
            return {
                "thread_id": thread_id,
                "approved": True,
                "result": result
            }
            
        except Exception as e:
            return {"error": f"Error approving transaction: {e}"}
    
    def reject_transaction(self, thread_id: str, reason: str = "") -> dict:
        """Reject a pending transaction."""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Get current state
            state = self.agent.get_state(config)
            
            if not state or not state.values:
                return {"error": f"No transaction found with ID: {thread_id}"}
            
            if not state.next:
                return {"error": f"Transaction #{thread_id} is not waiting for approval"}
            
            print(f"\n❌ MANAGER REJECTION: #{thread_id}")
            print(f"  Amount: ${state.values.get('amount', 0)}")
            print(f"  Reason: {reason}")
            
            # Update state with rejection
            updated_state = dict(state.values)
            updated_state["approval_granted"] = False
            updated_state["manager_notes"] = f"REJECTED: {reason}"
            updated_state["action_status"] = "REJECTED"
            
            # Resume execution (will go to END due to rejection)
            result = self.agent.invoke(updated_state, config)
            
            print(f"  Result: {result.get('action_status', 'Unknown')}")
            
            return {
                "thread_id": thread_id,
                "approved": False,
                "result": result
            }
            
        except Exception as e:
            return {"error": f"Error rejecting transaction: {e}"}
    
    def resume_transaction(self, thread_id: str) -> dict:
        """Resume a transaction without changing approval status."""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Get current state
            state = self.agent.get_state(config)
            
            if not state or not state.values:
                return {"error": f"No transaction found with ID: {thread_id}"}
            
            if not state.next:
                return {"error": f"Transaction #{thread_id} is not waiting"}
            
            print(f"\n▶️  RESUMING TRANSACTION: #{thread_id}")
            print(f"  Current status: {state.values.get('action_status', 'Unknown')}")
            print(f"  Waiting at node: {state.next}")
            
            # Resume execution with current state
            result = self.agent.invoke(state.values, config)
            
            print(f"  New status: {result.get('action_status', 'Unknown')}")
            
            return {
                "thread_id": thread_id,
                "resumed": True,
                "result": result
            }
            
        except Exception as e:
            return {"error": f"Error resuming transaction: {e}"}
    
    def list_transactions(self) -> dict:
        """List all active transactions."""
        
        print("\n📋 ACTIVE TRANSACTIONS:")
        print("="*60)
        
        if not self.active_threads:
            print("No active transactions")
            return {"count": 0, "transactions": []}
        
        transactions = []
        
        for thread_id, info in self.active_threads.items():
            state = info["state"]
            
            print(f"\n#{thread_id}:")
            print(f"  Amount: ${state.get('amount', 0)}")
            print(f"  Status: {state.get('action_status', 'Unknown')}")
            print(f"  Created: {info.get('created', 'Unknown')}")
            
            transactions.append({
                "thread_id": thread_id,
                "amount": state.get("amount", 0),
                "status": state.get("action_status", "Unknown"),
                "created": info.get("created", "")
            })
        
        print(f"\nTotal: {len(transactions)} transaction(s)")
        
        return {
            "count": len(transactions),
            "transactions": transactions
        }

# ============== 5. Test Suite ==============
def run_tests():
    """Run comprehensive tests of the transaction system."""
    
    print("🧪 TESTING TRANSACTION APPROVAL SYSTEM")
    print("=" * 70)
    
    # Build agent WITH interrupts
    print("\n🔧 Building agent with interrupt_before...")
    agent = build_transaction_agent(use_interrupt=True)
    
    # Create manager
    manager = TransactionManager(agent)
    
    # Test Scenario A: $50 transfer (auto-approved)
    print("\n" + "="*70)
    print("📋 TEST SCENARIO A: $50 Transfer (Should auto-approve)")
    print("="*70)
    
    result_a = manager.create_transaction(
        amount=50,
        transaction_type="transfer",
        recipient="John Doe",
        thread_id="test_a_50"
    )
    
    print(f"\n✅ Result: {result_a['result'].get('action_status', 'Unknown')}")
    
    # Check status
    status_a = manager.check_status("test_a_50")
    print(f"Final Status: {status_a.get('state', {}).get('action_status', 'Unknown')}")
    print(f"Waiting: {status_a.get('is_waiting', False)}")
    
    # Test Scenario B: $500 transfer (needs approval)
    print("\n\n" + "="*70)
    print("📋 TEST SCENARIO B: $500 Transfer (Needs manager approval)")
    print("="*70)
    
    result_b = manager.create_transaction(
        amount=500,
        transaction_type="wire_transfer",
        recipient="Jane Smith",
        thread_id="test_b_500"
    )
    
    print(f"\n⏸️  Result: {result_b['result'].get('action_status', 'Unknown')}")
    
    # Check status - should be waiting
    status_b = manager.check_status("test_b_500")
    print(f"Current Status: {status_b.get('state', {}).get('action_status', 'Unknown')}")
    print(f"Waiting: {status_b.get('is_waiting', False)}")
    
    if status_b.get('is_waiting'):
        print(f"⚠️  Waiting at node: {status_b.get('next', 'Unknown')}")
        
        # Simulate manager approval
        print("\n👔 SIMULATING MANAGER APPROVAL...")
        approval_result = manager.approve_transaction(
            thread_id="test_b_500",
            notes="Approved for quarterly bonus payment"
        )
        
        print(f"✅ Approval Result: {approval_result.get('approved', False)}")
        
        # Check final status
        final_status = manager.check_status("test_b_500")
        print(f"Final Status: {final_status.get('state', {}).get('action_status', 'Unknown')}")
    
    # Test Scenario C: $200 transfer (then reject)
    print("\n\n" + "="*70)
    print("📋 TEST SCENARIO C: $200 Transfer (Manager rejection)")
    print("="*70)
    
    result_c = manager.create_transaction(
        amount=200,
        transaction_type="vendor_payment",
        recipient="Suspicious Corp",
        thread_id="test_c_200"
    )
    
    print(f"\n⏸️  Initial Result: {result_c['result'].get('action_status', 'Unknown')}")
    
    # Check status - should be waiting
    status_c = manager.check_status("test_c_200")
    
    if status_c.get('is_waiting'):
        print(f"⚠️  Waiting for approval...")
        
        # Simulate manager rejection
        print("\n👔 SIMULATING MANAGER REJECTION...")
        reject_result = manager.reject_transaction(
            thread_id="test_c_200",
            reason="Suspicious recipient, requires further verification"
        )
        
        print(f"❌ Rejection Result: {not reject_result.get('approved', True)}")
        
        # Check final status
        final_status = manager.check_status("test_c_200")
        print(f"Final Status: {final_status.get('state', {}).get('action_status', 'Unknown')}")
    
    # List all transactions
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    manager.list_transactions()
    
    return manager

# ============== 6. Interactive Demo ==============
def interactive_demo():
    """Interactive demo of the transaction system."""
    
    print("💸 INTERACTIVE TRANSACTION APPROVAL SYSTEM")
    print("=" * 70)
    
    print("\nBuilding agent with interrupt_before on 'execute' node...")
    agent = build_transaction_agent(use_interrupt=True)
    manager = TransactionManager(agent)
    
    print("\nCommands:")
    print("  create <amount> <type> <recipient> - Create transaction")
    print("  status <thread_id> - Check transaction status")
    print("  approve <thread_id> [notes] - Approve transaction")
    print("  reject <thread_id> [reason] - Reject transaction")
    print("  resume <thread_id> - Resume transaction")
    print("  list - List all transactions")
    print("  quit - Exit")
    print("="*70)
    
    while True:
        command = input("\n> ").strip()
        
        if command.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif command.lower() == 'list':
            manager.list_transactions()
        elif command.startswith('create '):
            try:
                parts = command.split(' ', 3)
                if len(parts) < 4:
                    print("Usage: create <amount> <type> <recipient>")
                    continue
                
                amount = int(parts[1])
                tx_type = parts[2]
                recipient = parts[3]
                
                result = manager.create_transaction(amount, tx_type, recipient)
                print(f"Created transaction: {result.get('thread_id', 'Unknown')}")
                
            except ValueError:
                print("Error: Amount must be a number")
            except Exception as e:
                print(f"Error: {e}")
        
        elif command.startswith('status '):
            try:
                parts = command.split(' ', 1)
                if len(parts) < 2:
                    print("Usage: status <thread_id>")
                    continue
                
                thread_id = parts[1]
                manager.check_status(thread_id)
                
            except Exception as e:
                print(f"Error: {e}")
        
        elif command.startswith('approve '):
            try:
                parts = command.split(' ', 2)
                if len(parts) < 2:
                    print("Usage: approve <thread_id> [notes]")
                    continue
                
                thread_id = parts[1]
                notes = parts[2] if len(parts) > 2 else "Approved"
                
                result = manager.approve_transaction(thread_id, notes)
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"✅ Transaction approved!")
                
            except Exception as e:
                print(f"Error: {e}")
        
        elif command.startswith('reject '):
            try:
                parts = command.split(' ', 2)
                if len(parts) < 2:
                    print("Usage: reject <thread_id> [reason]")
                    continue
                
                thread_id = parts[1]
                reason = parts[2] if len(parts) > 2 else "Rejected by manager"
                
                result = manager.reject_transaction(thread_id, reason)
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"❌ Transaction rejected!")
                
            except Exception as e:
                print(f"Error: {e}")
        
        elif command.startswith('resume '):
            try:
                parts = command.split(' ', 1)
                if len(parts) < 2:
                    print("Usage: resume <thread_id>")
                    continue
                
                thread_id = parts[1]
                
                result = manager.resume_transaction(thread_id)
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"▶️  Transaction resumed!")
                
            except Exception as e:
                print(f"Error: {e}")
        
        else:
            print("Unknown command. Type 'quit' to exit.")

# ============== 7. DEEP DIVE: interrupt_before vs interrupt_after ==============
def study_interrupts():
    """Study the difference between interrupt_before and interrupt_after."""
    
    print("\n" + "="*70)
    print("🧠 DEEP DIVE: interrupt_before vs interrupt_after")
    print("="*70)
    
    print("\n📚 WHAT ARE INTERRUPTS?")
    print("-" * 40)
    print("""
    Interrupts in LangGraph allow you to pause execution
    before or after specific nodes, enabling human-in-the-loop workflows.
    
    Key concepts:
    • interrupt_before: Pause BEFORE executing the node
    • interrupt_after: Pause AFTER executing the node
    • Both allow manual inspection/intervention
    • Execution resumes with app.invoke()
    """)
    
    print("\n⚡ interrupt_before: Pause BEFORE Node Execution")
    print("-" * 40)
    print("""
    Use Case: Preventative approval
    Example: Manager approval before transaction execution
    
    Flow:
        [check] → [execute]  (would normally execute)
                   ↑
                PAUSED HERE (interrupt_before)
                Human reviews request
                Human approves/rejects
                Resume execution
    
    When to use:
    • Need to review/approve ACTION before it happens
    • Prevent unauthorized operations
    • Verify inputs before processing
    • Risk management scenarios
    """)
    
    print("\n🔍 interrupt_after: Pause AFTER Node Execution")
    print("-" * 40)
    print("""
    Use Case: Review/edit results
    Example: Human edits AI-generated content
    
    Flow:
        [generate_draft] → [send_to_client]  (would normally send)
                     ↓
                EXECUTES FIRST
                Produces draft
                   ↓
                PAUSED HERE (interrupt_after)
                Human reviews draft
                Human edits if needed
                Resume to send final version
    
    When to use:
    • Need to review OUTPUT before next step
    • Quality assurance of generated content
    • Fact-checking AI responses
    • Editing/refining automated work
    """)
    
    print("\n🎯 PRACTICAL SCENARIOS")
    print("-" * 40)
    
    print("\nScenario 1: Human verifies research before sending to client")
    print("""
    Question: Which interrupt to use?
    
    Analysis:
    • Research is generated by AI (node output)
    • Need to verify BEFORE sending to client
    • Want to review the research RESULTS
    
    Answer: interrupt_after on research node
    
    Why: 
    • Let AI generate research first
    • Then human reviews the output
    • Before it goes to the next node (sending to client)
    """)
    
    print("\nScenario 2: Human edits a draft the agent just wrote")
    print("""
    Question: Which interrupt to use?
    
    Analysis:
    • Agent writes draft (node output)  
    • Human needs to edit the draft
    • Before it's published/finalized
    
    Answer: interrupt_after on draft-writing node
    
    Why:
    • Agent produces draft first
    • Human edits the produced content
    • Then continues to publishing
    """)
    
    print("\nScenario 3: Manager approves expense before processing")
    print("""
    Question: Which interrupt to use?
    
    Analysis:
    • Expense request submitted
    • Need approval BEFORE any money moves
    • Prevent unauthorized transactions
    
    Answer: interrupt_before on payment-processing node
    
    Why:
    • Stop BEFORE payment happens
    • Human reviews the request
    • Approves or rejects
    • Then payment either happens or doesn't
    """)
    
    print("\n💡 KEY DIFFERENCES AT A GLANCE")
    print("-" * 40)
    print("""
    interrupt_before:
    • "Should we do this?"
    • Reviews the INTENT/REQUEST
    • Prevents unauthorized actions
    • Approval workflows
    
    interrupt_after:
    • "Is this output correct?"
    • Reviews the RESULT/OUTPUT
    • Quality control
    • Editing/refinement workflows
    """)
    
    print("\n🛠️ TECHNICAL IMPLEMENTATION")
    print("-" * 40)
    
    print("\ninterrupt_before example:")
    print("""
    workflow = StateGraph(State)
    workflow.add_node("process_payment", process_payment)
    
    # Will pause BEFORE process_payment executes
    app = workflow.compile(interrupt_before=["process_payment"])
    
    # When interrupted:
    # state.next = ('process_payment',)
    # Node hasn't run yet
    """)
    
    print("\ninterrupt_after example:")
    print("""
    workflow = StateGraph(State)
    workflow.add_node("generate_report", generate_report)
    
    # Will pause AFTER generate_report executes
    app = workflow.compile(interrupt_after=["generate_report"])
    
    # When interrupted:
    # state.next = (next_node_after_generate_report,)
    # generate_report HAS already run
    # You can review state['report_content']
    """)
    
    print("\n🎯 WHEN TO USE WHICH - DECISION TREE")
    print("-" * 40)
    print("""
    Ask: "Do I need to review what the agent WANTS to do?"
        ↓
        YES → interrupt_before
        ↓
        Review request/plan
        Approve/reject action
    
    Ask: "Do I need to review what the agent HAS DONE?"
        ↓
        YES → interrupt_after
        ↓
        Review output/results
        Edit/approve output
    """)

# ============== 8. Demonstration of Both Interrupt Types ==============
def demonstrate_both_interrupts():
    """Demonstrate both interrupt_before and interrupt_after."""
    
    print("\n" + "="*70)
    print("🎭 DEMONSTRATION: interrupt_before vs interrupt_after")
    print("="*70)
    
    # Example 1: interrupt_before
    print("\n1️⃣  EXAMPLE: interrupt_before (Preventative Approval)")
    print("-" * 40)
    
    class SimpleState(TypedDict):
        action: str
        approved: bool
        executed: bool
    
    def risky_action(state: SimpleState):
        print("⚠️  Executing risky action...")
        return {"executed": True, "action": "RISKY_ACTION_COMPLETED"}
    
    # Build with interrupt_before
    workflow_before = StateGraph(SimpleState)
    workflow_before.add_node("risky_action", risky_action)
    workflow_before.set_entry_point("risky_action")
    workflow_before.add_edge("risky_action", END)
    
    app_before = workflow_before.compile(interrupt_before=["risky_action"])
    
    print("""
    With interrupt_before:
    • Execution STOPS before 'risky_action'
    • Human reviews the request
    • Human decides: approve or reject
    • Only if approved, 'risky_action' executes
    """)
    
    # Example 2: interrupt_after  
    print("\n2️⃣  EXAMPLE: interrupt_after (Quality Review)")
    print("-" * 40)
    
    def generate_content(state: SimpleState):
        print("📝 Generating content...")
        return {"action": "CONTENT_GENERATED", "content": "Sample AI-generated text"}
    
    # Build with interrupt_after
    workflow_after = StateGraph(SimpleState)
    workflow_after.add_node("generate_content", generate_content)
    workflow_after.set_entry_point("generate_content")
    workflow_after.add_edge("generate_content", END)
    
    app_after = workflow_after.compile(interrupt_after=["generate_content"])
    
    print("""
    With interrupt_after:
    • 'generate_content' EXECUTES FIRST
    • Produces output
    • Then execution STOPS
    • Human reviews the generated content
    • Human can edit/approve
    • Then continues
    """)
    
    print("\n🎯 KEY TAKEAWAY:")
    print("-" * 40)
    print("""
    interrupt_before = "Should we do this?"
    • Reviews INTENT
    • Prevents bad actions
    
    interrupt_after = "Is this result good?"
    • Reviews OUTPUT  
    • Ensures quality
    
    Choose based on whether you need to review
    the REQUEST (before) or the RESULT (after).
    """)

# ============== 9. Main Execution ==============
if __name__ == "__main__":
    print("=" * 70)
    print("🏦 TRANSACTION APPROVAL SYSTEM WITH INTERRUPTS")
    print("=" * 70)
    
    print("\nFeatures:")
    print("• interrupt_before for manager approval")
    print("• Human-in-the-loop workflow")
    print("• Transaction tracking with thread IDs")
    print("• Approval/rejection workflows")
    
    print("\nOptions:")
    print("1. Run comprehensive tests")
    print("2. Interactive demo")
    print("3. Study interrupt_before vs interrupt_after")
    print("4. See both interrupt types demonstrated")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        manager = run_tests()
        
        print("\n" + "="*70)
        print("✅ TEST VERIFICATION")
        print("="*70)
        print("""
        ✅ REQUIREMENTS MET:
        
        1. State includes amount and action_status: ✓
        2. Node_Check determines if amount is high: ✓  
        3. Node_Execute prints "Transaction Successful": ✓
        4. interrupt_before on Node_Execute: ✓
        5. Scenario A ($50): Completes automatically: ✓
        6. Scenario B ($500): Stops for approval: ✓
        7. Resume with graph.invoke(): ✓
        8. Check state.next for waiting status: ✓
        
        🎯 KEY DEMONSTRATIONS:
        • $50 transaction auto-approved → completes
        • $500 transaction → pauses at interrupt
        • Manager approves → continues to execute
        • Manager rejects → ends without execution
        • state.next shows ('execute',) when waiting
        """)
        
    elif choice == "2":
        interactive_demo()
        
    elif choice == "3":
        study_interrupts()
        
    elif choice == "4":
        demonstrate_both_interrupts()
        
    else:
        print("Running comprehensive tests...")
        manager = run_tests()