"""
ROLE: WORKFLOW ENGINEER
JOB: "Deconstruct macro goals ('Evacuate Zone 7') into concrete,
tool-executable subtasks."

WHY THIS STEP EXISTS:
An agent can't act on a vague goal like "Evacuate Zone 7" directly. This
role breaks it into small, concrete steps the Agent Engineer's reasoning
loop can execute one at a time.
"""

def break_down_goal(zone, risk_level):
    """Turns one macro goal into an ordered list of concrete subtasks."""
    subtasks = [f"Check resource inventory for available ambulance near {zone}"]

    if risk_level == "Severe":
        subtasks.append(f"Reserve 1 ambulance and dispatch to {zone} IMMEDIATELY")
        subtasks.append(f"Check shelter capacity near {zone}")
        subtasks.append(f"Send evacuation alert to residents of {zone}")
    elif risk_level == "Moderate":
        subtasks.append(f"Put 1 ambulance on standby for {zone}")
        subtasks.append(f"Monitor {zone} sensors every 15 minutes")
    else:
        subtasks.append(f"No action needed for {zone}, continue monitoring")

    return subtasks


if __name__ == "__main__":
    # STEP: Demo breakdown of a macro goal from the mission brief
    goal_zone, goal_risk = "Zone 7", "Severe"
    tasks = break_down_goal(goal_zone, goal_risk)

    print(f"Macro Goal: 'Evacuate {goal_zone}' (risk={goal_risk})")
    print("Broken down into concrete subtasks:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")

    import json
    with open("data/subtasks.json", "w") as f:
        json.dump({"zone": goal_zone, "risk": goal_risk, "subtasks": tasks}, f, indent=2)
    print("\nSaved to data/subtasks.json")
