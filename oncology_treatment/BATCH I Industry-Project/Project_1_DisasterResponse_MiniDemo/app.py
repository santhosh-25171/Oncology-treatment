import streamlit as st
import os
import runpy
from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Disaster Response Command Center",
    page_icon="🚨",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent


# =========================================================
# LOAD STAGE SCRIPTS
# =========================================================

def load_script(rel):

    old = os.getcwd()
    folder = ROOT / Path(rel).parent

    try:
        os.chdir(folder)

        return runpy.run_path(
            str(ROOT / rel),
            run_name="dashboard_module"
        )

    finally:
        os.chdir(old)


@st.cache_resource
def load_stage_functions():

    # Stage 01 - Machine Learning
    s1 = load_script(
        "Stage01_ML/05_integration_engineer.py"
    )

    # Stage 03 - NLP
    s3 = load_script(
        "Stage03_NLP/05_integration_engineer.py"
    )

    # Stage 04 - SLM
    s4 = load_script(
        "Stage04_SLM/05_integration_engineer.py"
    )

    # Stage 05 - Generative AI
    s5 = load_script(
        "Stage05_GenAI/03_genai_engineer.py"
    )

    # Stage 06 - Agentic AI
    w = load_script(
        "Stage06_AgenticAI/02_workflow_engineer.py"
    )

    a = load_script(
        "Stage06_AgenticAI/03_agent_engineer.py"
    )

    return (
        s1["get_zone_risk_score"],
        s3["process_incoming_message"],
        s4["get_field_briefing"],
        s5["generate_scenario"],
        w["break_down_goal"],
        a["run_agent"]
    )


# =========================================================
# STAGE 02 - DEEP LEARNING
# =========================================================

def stage2_alert():

    folder = ROOT / "Stage02_DL"
    old = os.getcwd()

    try:

        os.chdir(folder)

        ns = runpy.run_path(
            str(folder / "05_integration_engineer.py"),
            run_name="dashboard_module"
        )

        test_image = np.load(
            "data/X_test.npy"
        )[0]

        return ns["get_flood_alert"](
            test_image,
            current_hour=6,
            current_level=4.2
        )

    finally:

        os.chdir(old)


# =========================================================
# LOAD RESOURCE INVENTORY
# =========================================================

def load_inventory():

    p = ROOT / "Stage06_AgenticAI/data/inventory.pkl"

    if p.exists():

        return pd.read_pickle(p)

    return pd.read_csv(
        ROOT / "Stage06_AgenticAI/data/resource_inventory.csv"
    )


# =========================================================
# AUTOMATIC EMERGENCY MESSAGE GENERATOR
# =========================================================

def generate_emergency_message(risk_level, zone):

    risk = risk_level.lower()

    if risk == "low":

        return (
            f"Minor water accumulation has been reported near {zone}. "
            f"No people are currently trapped. "
            f"Residents are advised to stay alert and monitor further weather updates."
        )

    elif risk in ["medium", "moderate"]:

        return (
            f"Water levels are rising near {zone}. "
            f"Some residents may require assistance. "
            f"Emergency teams are advised to prepare resources and monitor the situation."
        )

    else:

        return (
            f"Emergency! Water is rising rapidly near {zone}. "
            f"8 people are trapped and require immediate rescue. "
            f"Send a rescue boat immediately."
        )


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.title("🚨 Disaster Response Command Center")

st.caption(
    "End-to-End Multi-Agent AI System | "
    "ML → DL → NLP → SLM → GenAI → Agentic AI"
)


# =========================================================
# SIDEBAR - INCIDENT INPUT
# =========================================================

with st.sidebar:

    st.header("🚨 Incident Input")

    zone = st.selectbox(
        "Affected Zone",
        [f"Zone {i}" for i in range(1, 11)],
        index=1
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        0.0,
        200.0,
        10.0
    )

    gauge = st.number_input(
        "River Gauge Level (m)",
        0.0,
        10.0,
        2.0
    )

    calls = st.number_input(
        "Emergency Call Volume",
        0,
        100,
        2
    )

    roads = st.number_input(
        "Road Closures",
        0,
        20,
        0
    )

    override = st.toggle(
        "Enable Human Override",
        value=False
    )

    activate = st.button(
        "🚨 ACTIVATE AI RESPONSE",
        use_container_width=True,
        type="primary"
    )


# =========================================================
# WAIT UNTIL USER ACTIVATES SYSTEM
# =========================================================

if not activate:

    st.info(
        "👈 Enter an incident in the left panel "
        "and click **ACTIVATE AI RESPONSE**."
    )

    st.markdown("### System Flow")

    st.code(
        "Incident Input → ML Risk → DL Alert → "
        "NLP Intelligence → SLM Briefing → "
        "GenAI Scenario → Agent Dispatch → Final Rescue Plan"
    )

    st.stop()


# =========================================================
# LOAD ALL AI STAGES
# =========================================================

try:

    (
        risk_fn,
        nlp_fn,
        brief_fn,
        gen_fn,
        workflow_fn,
        agent_fn

    ) = load_stage_functions()


    # Stage 01 Prediction

    risk = risk_fn(
        rainfall,
        gauge,
        calls,
        roads
    )


except FileNotFoundError as e:

    st.error(
        "A generated model/data file is missing. "
        "Run the stage scripts first. Missing file: "
        + str(e)
    )

    st.stop()


except Exception as e:

    st.exception(e)

    st.stop()


st.success(
    "✅ Incident received. All AI stages are now connected."
)


# =========================================================
# STAGE 01 - MACHINE LEARNING
# =========================================================

st.header(
    "🟢 Stage 01 — Machine Learning: Zone Risk Score"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Risk Level",
    risk["risk_level"]
)

c2.metric(
    "Confidence",
    risk["confidence"]
)

c3.metric(
    "Zone",
    zone
)

st.info(
    "📋 Recommendation: "
    + risk["recommendation"]
)


# =========================================================
# AUTOMATIC AI-GENERATED MESSAGE
# =========================================================

message = generate_emergency_message(
    risk["risk_level"],
    zone
)

st.header(
    "📨 AI-Generated Emergency Incident Message"
)

st.write(message)


# =========================================================
# STAGE 02 - DEEP LEARNING
# =========================================================

st.header(
    "🔵 Stage 02 — Deep Learning: Flood & Trend Alert"
)

try:

    dl = stage2_alert()

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Camera Analysis",
        dl["camera_says"]
    )

    d2.metric(
        "Predicted Water Level in 3h",
        f"{dl['predicted_level_in_3h']} m"
    )

    d3.metric(
        "Unified Alert",
        dl["unified_alert"]
    )


except Exception as e:

    st.warning(
        "Stage 02 dashboard data could not be loaded: "
        + str(e)
    )

    dl = {
        "camera_says": "Unavailable",
        "predicted_level_in_3h": "Unavailable",
        "unified_alert": "Unavailable"
    }


# =========================================================
# STAGE 03 - NLP
# =========================================================

st.header(
    "🟠 Stage 03 — NLP: Emergency Message Intelligence"
)


# Process automatically generated message

nlp = nlp_fn(message)


# ---------------------------------------------------------
# STUDENT-FRIENDLY DISPLAY
# Internal NLP = Routine
# Dashboard Display = Normal
# ---------------------------------------------------------

urgency_display = (
    "Normal"
    if nlp["urgency"].lower() == "routine"
    else nlp["urgency"]
)


st.write(
    "**Automatically Generated Message:**",
    message
)


n1, n2, n3, n4 = st.columns(4)


# Display Normal instead of Routine

n1.metric(
    "Urgency",
    urgency_display
)


n2.metric(
    "Location",
    nlp["Location"]
)


n3.metric(
    "Headcount",
    nlp["Headcount"]
)


n4.metric(
    "Resource Needed",
    nlp["Resource Needed"]
)


# =========================================================
# STAGE 04 - SLM
# =========================================================

st.header(
    "🟣 Stage 04 — SLM: Commander Field Briefing"
)


full_report = (

    f"{zone} is experiencing "
    f"{risk['risk_level']} flood risk. "

    f"Rainfall is {rainfall} mm and "

    f"river gauge level is {gauge} m. "

    f"Emergency call volume is {calls} "

    f"with {roads} road closures. "

    f"Incoming emergency report: {message}. "

    f"Stage 02 camera result is "
    f"{dl['camera_says']}. "

    f"Predicted water level in 3 hours is "
    f"{dl['predicted_level_in_3h']}. "

    f"Immediate recommendation is: "
    f"{risk['recommendation']}"
)


brief = brief_fn(
    full_report
)


st.warning(
    "🎙️ **5-Second Commander Briefing:** "
    + brief
)


# =========================================================
# STAGE 05 - GENERATIVE AI
# =========================================================

st.header(
    "🟡 Stage 05 — Generative AI: Synthetic Disaster Stress Test"
)


try:

    scenarios = gen_fn(
        risk["risk_level"],
        n=3
    )


    for i, scenario in enumerate(
        scenarios,
        1
    ):

        with st.expander(
            f"🌪️ Synthetic Disaster Scenario {i}",
            expanded=(i == 1)
        ):

            for key, value in scenario.items():

                st.write(
                    f"**{key.replace('_', ' ').title()}:** {value}"
                )


    st.caption(
        "Generated scenarios are used to stress-test "
        "the disaster response system."
    )


except Exception as e:

    st.warning(
        "Synthetic scenario generation unavailable: "
        + str(e)
    )


# =========================================================
# STAGE 06 - AGENTIC AI
# =========================================================

st.header(
    "🔴 Stage 06 — Agentic AI: Autonomous Dispatch Engine"
)


inventory = load_inventory().copy()


if override:

    st.error(
        "⏸️ HUMAN OVERRIDE ACTIVE — "
        "Agent is paused. Commander approval is required."
    )

    st.write({
        "status": "PAUSED_FOR_HUMAN",
        "zone": zone
    })


    col1, col2, col3 = st.columns(3)


    col1.button(
        "✅ Approve Plan",
        disabled=True
    )


    col2.button(
        "✏️ Modify Plan",
        disabled=True
    )


    col3.button(
        "❌ Cancel Dispatch",
        disabled=True
    )


    final_priority = "Paused"


else:

    # Convert headcount safely to number

    try:

        people = int(
            nlp["Headcount"]
        )

    except:

        people = 0


    # -----------------------------------------------------
    # AGENT DECISION
    #
    # High priority if:
    # NLP says Urgent OR people are trapped
    # -----------------------------------------------------

    human_emergency = (

        nlp["urgency"].lower() == "urgent"

        or

        people > 0
    )


    if human_emergency:

        final_priority = "HIGH"

        st.metric(
            "Final Agent Priority",
            final_priority
        )

        st.metric(
            "People Requiring Help",
            people
        )


        st.warning(
            "⚠️ MULTI-SOURCE EMERGENCY DETECTED\n\n"
            f"ML reports {risk['risk_level']} sensor-based risk, "
            f"but NLP detected {nlp['urgency']} urgency with "
            f"{people} person(s) requiring help. "
            "The agent prioritizes the human emergency."
        )


        subtasks = [

            f"Assess immediate human emergency in {zone}",

            f"Prioritize rescue for {people} person(s) "
            f"needing help in {zone}",

            f"Coordinate requested resource: "
            f"{nlp['Resource Needed']}",

            f"Check available ambulance and rescue resources "
            f"near {zone}",

            f"Notify emergency command and prepare shelter "
            f"support for {zone}",

            f"Monitor changing flood conditions in {zone}"

        ]


    else:

        # No immediate emergency
        # Agent follows ML risk

        final_priority = risk["risk_level"]

        st.metric(
            "Final Agent Priority",
            final_priority
        )

        st.metric(
            "People Requiring Help",
            people
        )


        st.info(
            "No immediate human emergency was detected. "
            "The agent follows the ML risk assessment."
        )


        subtasks = workflow_fn(
            zone,
            risk["risk_level"]
        )


    # -----------------------------------------------------
    # DISPLAY RESCUE PLAN
    # -----------------------------------------------------

    trace = agent_fn(
        subtasks,
        inventory
    )


    st.subheader(
        "🤖 Autonomous Rescue Plan"
    )


    for i, task in enumerate(
        subtasks,
        1
    ):

        st.write(
            f"{i}. {task}"
        )


    # -----------------------------------------------------
    # REASONING TRACE
    # -----------------------------------------------------

    st.subheader(
        "🧠 Transparent Agent Reasoning Trace"
    )


    for line in trace:

        st.code(
            line
        )


    st.success(
        f"🚑 RESPONSE PLAN COMPLETED — "
        f"{len(trace)} reasoning/action events executed "
        f"for {zone}."
    )


# =========================================================
# FINAL OUTCOME
# =========================================================

st.divider()


st.header(
    "🏁 FINAL OUTCOME"
)


# Student-friendly final urgency display

final_urgency_display = (
    "Normal"
    if nlp["urgency"].lower() == "routine"
    else nlp["urgency"]
)


st.markdown(
    f"""
### 🚨 Rescue Decision Summary

**Affected Zone:** {zone}

**ML Risk:** {risk['risk_level']}
({risk['confidence']} confidence)

**NLP Urgency:** {final_urgency_display}

**People Identified:** {nlp['Headcount']}

**Resource Requested:** {nlp['Resource Needed']}

**Final Agent Priority:** {final_priority}

**Agent Status:** {'PAUSED FOR HUMAN OVERRIDE' if override else 'AUTONOMOUS RESPONSE COMPLETED'}
"""
)


st.caption(
    "This dashboard is the visual integration layer that connects "
    "all six AI stages into one final capstone demonstration."
)