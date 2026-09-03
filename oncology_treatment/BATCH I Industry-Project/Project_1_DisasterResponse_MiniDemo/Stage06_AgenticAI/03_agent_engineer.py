"""
ROLE: AGENT ENGINEER
JOB: "Design the core reasoning loop, tool invocation parameters, and
decision logic."

WHY THIS STEP EXISTS:
This is the heart of Agentic AI - a REASON -> ACT -> OBSERVE loop (the
"ReAct" pattern mentioned in the study material). The agent looks at the
situation, decides an action, "does" it (calls a tool), observes the
result, and repeats until the goal is done.

SIMPLIFIED FOR CLASSROOM: A real agent uses an LLM to generate each
reasoning step. Here we hand-code the same LOOP STRUCTURE with simple
rules, so students see exactly how the loop works.
"""

import pandas as pd
import json

inventory = pd.read_pickle("data/inventory.pkl")
with open("data/subtasks.json") as f:
    plan = json.load(f)


def run_agent(subtasks, inventory):
    """The REASON -> ACT -> OBSERVE loop."""
    trace = []
    for step_num, task in enumerate(subtasks, 1):
        # REASON: decide what this task requires
        trace.append(f"[REASON] Step {step_num}: Need to '{task}'")

        # ACT: call a "tool" (here: check/update the inventory dataframe)
        if "ambulance" in task.lower() and "dispatch" in task.lower():
            available = inventory[(inventory["resource_type"] == "Ambulance") &
                                   (inventory["status"] == "Available")]
            if not available.empty:
                chosen = available.iloc[0]
                inventory.loc[inventory["resource_id"] == chosen["resource_id"], "status"] = "Dispatched"
                trace.append(f"[ACT] Tool call: dispatch_ambulance({chosen['resource_id']})")
                trace.append(f"[OBSERVE] {chosen['resource_id']} successfully dispatched.")
            else:
                trace.append("[ACT] Tool call: dispatch_ambulance() -> FAILED")
                trace.append("[OBSERVE] No ambulance available! Escalate to human commander.")
        else:
            trace.append(f"[ACT] Tool call: log_action('{task}')")
            trace.append("[OBSERVE] Logged successfully.")

    return trace


if __name__ == "__main__":
    print(f"=== Agent running plan for {plan['zone']} (risk={plan['risk']}) ===\n")
    reasoning_trace = run_agent(plan["subtasks"], inventory)
    for line in reasoning_trace:
        print(line)

    inventory.to_pickle("data/inventory_after.pkl")
    with open("data/reasoning_trace.json", "w") as f:
        json.dump(reasoning_trace, f, indent=2)
    print("\nSaved full reasoning trace to data/reasoning_trace.json")
