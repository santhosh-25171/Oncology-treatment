# Stage 06 — Agentic AI (Mini Demo)

**Mission:** Bring everything together — an autonomous agent that reads
ML risk scores, DL vision alerts, and NLP/SLM intelligence, then decides
its own rescue plan and dispatches resources, with a transparent reasoning
trace and a human override option.

**Simplified for classroom pace:** A real agent uses an LLM with tool-calling
in a reasoning loop (the "ReAct" pattern: Reason -> Act -> Observe -> repeat).
Here we hand-code that SAME loop with simple if/else rules, so students see
the loop structure and the reasoning trace clearly, without needing an LLM API.

## Run order

| Script | Role | What it does |
|---|---|---|
| `01_knowledge_engineer.py` | Knowledge Engineer | Builds the SOP knowledge base + resource inventory |
| `02_workflow_engineer.py` | Workflow Engineer | Breaks a macro goal ("Evacuate Zone 7") into sub-tasks |
| `03_agent_engineer.py` | Agent Engineer | Runs the reasoning loop that decides dispatch actions |
| `04_evaluation_engineer.py` | Evaluation Engineer | Tests the agent on a hard case: 2 zones, 1 ambulance |
| `05_integration_engineer.py` | Integration Engineer | Command-center style function with human override control |
