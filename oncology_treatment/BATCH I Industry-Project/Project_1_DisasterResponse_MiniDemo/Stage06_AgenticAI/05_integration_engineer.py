"""
ROLE: INTEGRATION ENGINEER (Agentic AI stage)
JOB: "Build the Orchestration Command Center UI featuring live execution
monitoring and manual override."

WHY THIS STEP EXISTS:
Wraps the whole agent loop into ONE command-center style function, with a
human override switch - matching the problem statement's requirement for
a "Human Override Control".
"""

import pandas as pd
import json


def run_command_center(zone, risk_level, subtasks_fn, agent_fn, inventory, human_override=False):
    """This IS the command center's core function."""
    print(f"=== COMMAND CENTER: Incoming incident at {zone} (risk={risk_level}) ===")

    if human_override:
        print("[HUMAN OVERRIDE ACTIVE] Agent paused. Awaiting manual commander decision.")
        return {"status": "PAUSED_FOR_HUMAN", "zone": zone}

    subtasks = subtasks_fn(zone, risk_level)
    print(f"Plan generated: {len(subtasks)} subtasks")

    trace = agent_fn(subtasks, inventory)
    print("\nLive reasoning trace:")
    for line in trace:
        print(f"  {line}")

    return {"status": "COMPLETED", "zone": zone, "steps_executed": len(trace)}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from importlib import import_module

    workflow_mod = import_module("02_workflow_engineer".replace("-", "_")) if False else None
    # Simple re-import via exec since filenames start with numbers (not valid module names)
    ns = {}
    exec(open("02_workflow_engineer.py").read().split("if __name__")[0], ns)
    exec(open("03_agent_engineer.py").read().split("if __name__")[0], ns)

    inventory = pd.read_pickle("data/inventory.pkl")

    print("\n--- DEMO 1: Normal run ---")
    result1 = run_command_center("Zone 7", "Severe", ns["break_down_goal"], ns["run_agent"],
                                  inventory.copy(), human_override=False)
    print("\nResult:", result1)

    print("\n--- DEMO 2: Human override active ---")
    result2 = run_command_center("Zone 3", "Severe", ns["break_down_goal"], ns["run_agent"],
                                  inventory.copy(), human_override=True)
    print("\nResult:", result2)
