"""
Interactive Web Dashboard for Stage 5 Synthetic Oncology Scenario Testing.
Renders an integrated clinical stress-testing dashboard with KPI cards,
filtering, live inspection, single & batch evaluation execution, and longitudinal history.
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .api import app, service

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stage 5 - Synthetic Scenario Testing Dashboard</title>
  <style>
    :root {
      --bg-color: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-green: #34d399;
      --accent-amber: #fbbf24;
      --accent-rose: #f43f5e;
      --accent-purple: #c084fc;
      --table-hover: #283548;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg-color); color: var(--text-main); line-height: 1.5; padding: 24px; }
    .banner-warning {
      background: linear-gradient(90deg, #b91c1c 0%, #991b1b 100%);
      color: #fff;
      padding: 12px 20px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 4px 12px rgba(185, 28, 28, 0.3);
      margin-bottom: 24px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 16px;
    }
    .header h1 { font-size: 26px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }
    .header .subtitle { color: var(--text-muted); font-size: 14px; margin-top: 4px; }
    .header-actions { display: flex; gap: 12px; }
    .btn {
      background-color: var(--accent-blue);
      color: #0f172a;
      border: none;
      padding: 10px 18px;
      font-size: 13px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .btn:hover { filter: brightness(1.1); transform: translateY(-1px); }
    .btn-secondary { background-color: var(--card-border); color: #fff; }
    .btn-secondary:hover { background-color: #475569; }
    .btn-accent { background-color: var(--accent-purple); color: #0f172a; }
    
    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .kpi-card {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
    }
    .kpi-title { font-size: 12px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; }
    .kpi-value { font-size: 28px; font-weight: 800; margin-top: 4px; color: #fff; }
    .kpi-subtext { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
    .text-pass { color: var(--accent-green) !important; }
    .text-review { color: var(--accent-amber) !important; }
    .text-fail { color: var(--accent-rose) !important; }
    .text-stress { color: var(--accent-purple) !important; }
    .text-realism { color: var(--accent-blue) !important; }

    /* Category pills */
    .category-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
      align-items: center;
    }
    .cat-label { font-size: 12px; font-weight: 600; color: var(--text-muted); margin-right: 4px; }
    .badge-pill {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .badge-pill.active, .badge-pill:hover {
      border-color: var(--accent-blue);
      color: var(--accent-blue);
      background-color: rgba(56, 189, 248, 0.1);
    }

    /* Filters Bar */
    .filter-bar {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      margin-bottom: 24px;
    }
    .filter-group { display: flex; align-items: center; gap: 8px; }
    .filter-label { font-size: 12px; font-weight: 600; color: var(--text-muted); }
    .filter-select, .filter-input {
      background-color: var(--bg-color);
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 13px;
      outline: none;
    }
    .filter-select:focus, .filter-input:focus { border-color: var(--accent-blue); }

    /* Main Grid layout */
    .dashboard-layout {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
    }
    @media (max-width: 1080px) {
      .dashboard-layout { grid-template-columns: 1fr; }
    }

    /* Panels */
    .panel {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 20px;
      height: 100%;
      display: flex;
      flex-direction: column;
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
    }
    .panel-title { font-size: 16px; font-weight: 700; color: #fff; }

    /* Table */
    .table-container { overflow-x: auto; max-height: 520px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
    th {
      background-color: #182234;
      color: var(--text-muted);
      font-weight: 600;
      padding: 10px 12px;
      border-bottom: 1px solid var(--card-border);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(51, 65, 85, 0.5);
      color: var(--text-main);
    }
    tr.scenario-row { cursor: pointer; transition: background 0.15s; }
    tr.scenario-row:hover { background-color: var(--table-hover); }
    tr.scenario-row.selected { background-color: rgba(56, 189, 248, 0.15); border-left: 3px solid var(--accent-blue); }

    .status-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.3px;
    }
    .badge-pass { background-color: rgba(52, 211, 153, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
    .badge-review { background-color: rgba(251, 191, 36, 0.15); color: var(--accent-amber); border: 1px solid var(--accent-amber); }
    .badge-fail { background-color: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid var(--accent-rose); }
    .badge-not-eval { background-color: rgba(148, 163, 184, 0.15); color: var(--text-muted); border: 1px solid var(--text-muted); }

    .tag-mono { font-family: monospace; font-size: 12px; background: #0f172a; padding: 2px 6px; border-radius: 4px; }

    /* Inspector Card */
    .detail-card {
      background-color: #131d2e;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 14px;
      margin-bottom: 14px;
    }
    .detail-heading { font-size: 12px; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }
    .field-row { display: flex; margin-bottom: 6px; font-size: 13px; }
    .field-label { width: 140px; color: var(--text-muted); font-weight: 500; }
    .field-value { flex: 1; color: #fff; }

    .audit-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .audit-item {
      background: #0f172a;
      border: 1px solid var(--card-border);
      padding: 8px 12px;
      border-radius: 4px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
    }

    /* History Table */
    .history-box {
      margin-top: 24px;
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 20px;
    }
  </style>
</head>
<body>

  <!-- Synthetic Warning Banner -->
  <div class="banner-warning">
    <span>⚠️ CRITICAL SAFETY NOTICE:</span>
    <span>SYNTHETIC TEST SCENARIOS ONLY (Stage 5 GenAI) — Strictly for stress-testing decision logic. NEVER treat as real patient records or clinical recommendations.</span>
  </div>

  <!-- Header -->
  <div class="header">
    <div>
      <h1>Personalized Precision Oncology <span style="font-size: 16px; background: #334155; padding: 2px 8px; border-radius: 4px; color: var(--accent-blue);">Stage 5</span></h1>
      <div class="subtitle">Synthetic Stress-Test Scenario Testing & Agent Decision-Logic Evaluation Dashboard</div>
    </div>
    <div class="header-actions">
      <button class="btn btn-secondary" onclick="reloadScenarios()">↻ Reload Files</button>
      <button class="btn btn-accent" onclick="runBatchEvaluation()">⚡ Batch Evaluate (Run All)</button>
    </div>
  </div>

  <!-- KPIs -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-title">Total Scenarios</div>
      <div class="kpi-value" id="kpi-total">-</div>
      <div class="kpi-subtext">All Verified Synthetic</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Audited PASS</div>
      <div class="kpi-value text-pass" id="kpi-pass">-</div>
      <div class="kpi-subtext">Ready for agent testing</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Audited REVIEW</div>
      <div class="kpi-value text-review" id="kpi-review">-</div>
      <div class="kpi-subtext">High uncertainty/sparse</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Audited FAIL</div>
      <div class="kpi-value text-fail" id="kpi-fail">0</div>
      <div class="kpi-subtext">Schema or logic faults</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Avg Decision Stress</div>
      <div class="kpi-value text-stress" id="kpi-stress">-</div>
      <div class="kpi-subtext">Out of 5.0 (Extreme)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Avg Realism</div>
      <div class="kpi-value text-realism" id="kpi-realism">-</div>
      <div class="kpi-subtext">Out of 5.0 (Reference)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Blind Spot Target Coverage</div>
      <div class="kpi-value" id="kpi-coverage">-</div>
      <div class="kpi-subtext">Covered EDA targets (not pass rate)</div>
    </div>
  </div>

  <!-- Category Pills -->
  <div class="category-bar">
    <span class="cat-label">Categories:</span>
    <span class="badge-pill active" onclick="filterByCategory('')">All Categories</span>
    <span class="badge-pill" onclick="filterByCategory('rare_mutation')">rare_mutation</span>
    <span class="badge-pill" onclick="filterByCategory('compound_resistance')">compound_resistance</span>
    <span class="badge-pill" onclick="filterByCategory('bypass_resistance')">bypass_resistance</span>
    <span class="badge-pill" onclick="filterByCategory('conflicting_biomarkers')">conflicting_biomarkers</span>
    <span class="badge-pill" onclick="filterByCategory('unobserved_fusions')">unobserved_fusions</span>
    <span class="badge-pill" onclick="filterByCategory('lineage_switch')">lineage_switch</span>
    <span class="badge-pill" onclick="filterByCategory('wildcard_extreme')">wildcard_extreme</span>
  </div>

  <!-- Filters Bar -->
  <div class="filter-bar">
    <div class="filter-group">
      <span class="filter-label">Status:</span>
      <select class="filter-select" id="filter-status" onchange="applyFilters()">
        <option value="">All Statuses</option>
        <option value="PASS">PASS</option>
        <option value="REVIEW">REVIEW</option>
        <option value="FAIL">FAIL</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Uncertainty:</span>
      <select class="filter-select" id="filter-unc" onchange="applyFilters()">
        <option value="">All Uncertainty</option>
        <option value="low">Low</option>
        <option value="moderate">Moderate</option>
        <option value="high">High</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Method:</span>
      <select class="filter-select" id="filter-method" onchange="applyFilters()">
        <option value="">All Methods</option>
        <option value="template">Template</option>
        <option value="llm">LLM</option>
      </select>
    </div>
    <div class="filter-group" style="flex: 1; justify-content: flex-end;">
      <span class="filter-label">Search:</span>
      <input type="text" class="filter-input" id="search-input" placeholder="Search ID or Blind Spot..." oninput="applyFilters()" style="min-width: 220px;">
    </div>
  </div>

  <!-- Layout: Table on Left, Details on Right -->
  <div class="dashboard-layout">
    <!-- Scenarios Table -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Synthetic Stress-Test Scenarios</div>
        <div style="font-size: 12px; color: var(--text-muted);" id="scenario-count-label">Loading...</div>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Scenario ID</th>
              <th>Category</th>
              <th>Blind Spot</th>
              <th>Uncertainty</th>
              <th>Stress</th>
              <th>Realism</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="scenario-table-body">
            <!-- Dynamic rows -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Inspector Panel -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Scenario Inspector & Live Audit</div>
        <div id="inspector-actions">
          <button class="btn" style="padding: 6px 12px; font-size: 12px;" onclick="evaluateCurrentScenario()">⚡ Evaluate Now</button>
        </div>
      </div>
      <div id="inspector-content" style="overflow-y: auto; max-height: 520px; padding-right: 4px;">
        <div style="text-align: center; color: var(--text-muted); padding: 40px;">Select a scenario from the table to inspect details.</div>
      </div>
    </div>
  </div>

  <!-- Longitudinal History Section -->
  <div class="history-box">
    <div class="panel-header">
      <div class="panel-title">Synthetic Scenario Evaluation Runs (Append-Only Log)</div>
      <div style="font-size: 12px; color: var(--text-muted);">Tracks repeated synthetic stress-test runs, agent decision-logic evaluations, and status transitions</div>
    </div>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Scenario ID</th>
            <th>Target Blind Spot</th>
            <th>Decision Stress</th>
            <th>Realism</th>
            <th>Status</th>
            <th>Timestamp (UTC)</th>
          </tr>
        </thead>
        <tbody id="history-table-body">
          <!-- Dynamic history rows -->
        </tbody>
      </table>
    </div>
  </div>

  <script>
    let allScenarios = [];
    let currentCategory = "";
    let selectedScenarioId = null;

    async function init() {
      await loadSummary();
      await loadScenarios();
      await loadHistory();
    }

    async function loadSummary() {
      try {
        const res = await fetch("/stage5/evaluation/summary");
        if (!res.ok) return;
        const data = await res.json();
        document.getElementById("kpi-total").innerText = data.total_scenarios ?? 0;
        document.getElementById("kpi-pass").innerText = data.passed ?? 0;
        document.getElementById("kpi-review").innerText = data.review ?? 0;
        document.getElementById("kpi-fail").innerText = data.failed ?? 0;
        document.getElementById("kpi-stress").innerText = data.average_stress_score ? data.average_stress_score.toFixed(2) : "-";
        document.getElementById("kpi-realism").innerText = data.average_realism_score ? data.average_realism_score.toFixed(2) : "-";
        if (data.blind_spot_coverage_display) {
          document.getElementById("kpi-coverage").innerText = `${data.blind_spot_coverage ?? 0} / ${data.total_blind_spot_targets ?? 24} (${data.blind_spot_coverage_pct ?? 75.0}%)`;
        } else {
          document.getElementById("kpi-coverage").innerText = `${data.blind_spot_coverage ?? 0} / 24 targets (75.0%)`;
        }
      } catch (err) {
        console.error("Failed to load summary", err);
      }
    }

    async function loadScenarios() {
      try {
        const res = await fetch("/stage5/scenarios");
        if (!res.ok) return;
        allScenarios = await res.json();
        applyFilters();
        if (allScenarios.length > 0 && !selectedScenarioId) {
          selectScenario(allScenarios[0].scenario_id);
        }
      } catch (err) {
        console.error("Failed to load scenarios", err);
      }
    }

    function filterByCategory(cat) {
      currentCategory = cat;
      document.querySelectorAll(".badge-pill").forEach(p => {
        if ((cat === "" && p.innerText === "All Categories") || p.innerText === cat) {
          p.classList.add("active");
        } else {
          p.classList.remove("active");
        }
      });
      applyFilters();
    }

    function applyFilters() {
      const statusFilter = document.getElementById("filter-status").value;
      const uncFilter = document.getElementById("filter-unc").value;
      const methodFilter = document.getElementById("filter-method").value;
      const searchTerm = document.getElementById("search-input").value.toLowerCase().trim();

      const filtered = allScenarios.filter(s => {
        if (currentCategory && s.scenario_category !== currentCategory) return false;
        if (statusFilter && s.evaluation_status !== statusFilter) return false;
        if (uncFilter && s.uncertainty_level !== uncFilter) return false;
        if (methodFilter && s.generation_method !== methodFilter) return false;
        if (searchTerm) {
          const matchId = s.scenario_id.toLowerCase().includes(searchTerm);
          const matchBs = s.target_blind_spot.toLowerCase().includes(searchTerm);
          const matchCat = s.scenario_category.toLowerCase().includes(searchTerm);
          if (!matchId && !matchBs && !matchCat) return false;
        }
        return true;
      });

      renderTable(filtered);
    }

    function renderTable(scenarios) {
      const tbody = document.getElementById("scenario-table-body");
      tbody.innerHTML = "";
      document.getElementById("scenario-count-label").innerText = `Showing ${scenarios.length} of ${allScenarios.length} scenarios`;

      scenarios.forEach(sc => {
        const tr = document.createElement("tr");
        tr.className = `scenario-row ${sc.scenario_id === selectedScenarioId ? 'selected' : ''}`;
        tr.onclick = () => selectScenario(sc.scenario_id);

        let badgeClass = "badge-not-eval";
        if (sc.evaluation_status === "PASS") badgeClass = "badge-pass";
        else if (sc.evaluation_status === "REVIEW") badgeClass = "badge-review";
        else if (sc.evaluation_status === "FAIL") badgeClass = "badge-fail";

        const stressDisplay = typeof sc.decision_stress_score === 'number' ? sc.decision_stress_score.toFixed(2) : sc.decision_stress_score;
        const realismDisplay = typeof sc.realism_score === 'number' ? sc.realism_score.toFixed(2) : sc.realism_score;

        tr.innerHTML = `
          <td><span class="tag-mono">${sc.scenario_id}</span></td>
          <td>${sc.scenario_category}</td>
          <td><span class="tag-mono" style="color: var(--accent-amber);">${sc.target_blind_spot}</span></td>
          <td style="text-transform: capitalize;">${sc.uncertainty_level}</td>
          <td><strong style="color: var(--accent-purple);">${stressDisplay}</strong></td>
          <td><strong style="color: var(--accent-blue);">${realismDisplay}</strong></td>
          <td><span class="status-badge ${badgeClass}">${sc.evaluation_status}</span></td>
          <td>
            <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="event.stopPropagation(); liveEvaluateRow('${sc.scenario_id}')">Audit</button>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function selectScenario(scenarioId) {
      selectedScenarioId = scenarioId;
      document.querySelectorAll(".scenario-row").forEach(r => r.classList.remove("selected"));
      applyFilters();

      const container = document.getElementById("inspector-content");
      container.innerHTML = `<div style="text-align: center; padding: 20px; color: var(--text-muted);">Loading details for ${scenarioId}...</div>`;

      try {
        const res = await fetch(`/stage5/scenarios/${scenarioId}`);
        if (!res.ok) throw new Error("Scenario not found");
        const data = await res.json();
        renderInspector(data);
      } catch (err) {
        container.innerHTML = `<div style="color: var(--accent-rose); padding: 20px;">Failed to load scenario: ${err.message}</div>`;
      }
    }

    function renderInspector(sc) {
      const c = document.getElementById("inspector-content");
      const evalData = sc.evaluation || {};

      let statusBadge = `<span class="status-badge badge-not-eval">NOT EVALUATED</span>`;
      if (evalData.evaluation_status === "PASS") statusBadge = `<span class="status-badge badge-pass">AUDITED: PASS</span>`;
      else if (evalData.evaluation_status === "REVIEW") statusBadge = `<span class="status-badge badge-review">AUDITED: REVIEW</span>`;
      else if (evalData.evaluation_status === "FAIL") statusBadge = `<span class="status-badge badge-fail">AUDITED: FAIL</span>`;

      // Genomic alterations list
      const alterationsHtml = (sc.genomic_profile.alterations || []).map(a => 
        `<div><strong>${a.gene}</strong> <span class="tag-mono">${a.variant}</span> - <span style="font-size:11px; color: var(--text-muted);">${a.alteration_type} (${a.status})</span></div>`
      ).join("") || "None";

      const cooccHtml = (sc.genomic_profile.cooccurring_alterations || []).map(a => 
        `<div><strong>${a.gene}</strong> <span class="tag-mono">${a.variant}</span> (${a.status})</div>`
      ).join("") || "None";

      // Sub-audit breakdown
      const sValid = evalData.schema_validation?.valid ? "PASS" : (evalData.schema_validation?.valid === false ? "FAIL" : "N/A");
      const pValid = evalData.provenance_validation?.valid ? "PASS" : (evalData.provenance_validation?.valid === false ? "FAIL" : "N/A");
      const bsValid = evalData.blind_spot_audit?.blind_spot_supported ? "PASS" : (evalData.blind_spot_audit?.blind_spot_supported === false ? "FAIL" : "N/A");
      const gValid = evalData.genomic_consistency?.valid ? "PASS" : (evalData.genomic_consistency?.valid === false ? "FAIL" : "N/A");
      const clValid = evalData.clinical_consistency?.valid ? "PASS" : (evalData.clinical_consistency?.valid === false ? "FAIL" : "N/A");

      c.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <div>
            <h2 style="font-size: 18px; color: #fff;">${sc.scenario_id}</h2>
            <span style="font-size: 11px; color: var(--accent-rose); font-weight: 700;">${sc.synthetic_badge}</span>
          </div>
          <div>${statusBadge}</div>
        </div>

        <!-- Demographics & Clinical -->
        <div class="detail-card">
          <div class="detail-heading">Patient Demographics & Context</div>
          <div class="field-row"><div class="field-label">Age & Sex:</div><div class="field-value">${sc.patient_context.age_group}, ${sc.patient_context.sex}</div></div>
          <div class="field-row"><div class="field-label">Diagnosis:</div><div class="field-value">${sc.patient_context.cancer_type} - ${sc.patient_context.histology} (${sc.patient_context.stage})</div></div>
          <div class="field-row"><div class="field-label">Disease Status:</div><div class="field-value">${sc.clinical_context.disease_status}</div></div>
          <div class="field-row"><div class="field-label">Prior Therapy:</div><div class="field-value">${sc.patient_context.prior_treatment_context}</div></div>
          <div class="field-row"><div class="field-label">Progression:</div><div class="field-value">${sc.clinical_context.progression_context}</div></div>
        </div>

        <!-- Genomics & Biomarkers -->
        <div class="detail-card">
          <div class="detail-heading">Genomic Profile & Biomarkers</div>
          <div class="field-row"><div class="field-label">Primary Alterations:</div><div class="field-value">${alterationsHtml}</div></div>
          <div class="field-row"><div class="field-label">Co-occurring:</div><div class="field-value">${cooccHtml}</div></div>
          <div class="field-row"><div class="field-label">Biomarkers:</div><div class="field-value">TMB: <strong>${sc.biomarkers.tmb}</strong> (${sc.biomarkers.tmb_status}) | PD-L1 TPS: <strong>${sc.biomarkers.pdl1_tps}%</strong> | MSI: <strong>${sc.biomarkers.msi_status}</strong></div></div>
        </div>

        <!-- Blind Spot & Assumptions -->
        <div class="detail-card">
          <div class="detail-heading">Target Blind Spot & Synthetic Rationale</div>
          <div class="field-row"><div class="field-label">Blind Spot:</div><div class="field-value"><strong style="color: var(--accent-amber);">${sc.target_blind_spot.blind_spot_id}</strong> (${sc.target_blind_spot.category})</div></div>
          <div class="field-row"><div class="field-label">Target Reason:</div><div class="field-value">${sc.target_blind_spot.reason}</div></div>
          <div class="field-row"><div class="field-label">Synthetic Assumptions:</div><div class="field-value">${sc.synthetic_assumptions.join("; ")}</div></div>
          <div class="field-row"><div class="field-label">Reference Evidence:</div><div class="field-value">${sc.reference_evidence.join("; ")}</div></div>
          <div class="field-row"><div class="field-label">Uncertainty:</div><div class="field-value"><span style="text-transform: capitalize; font-weight: 600;">${sc.uncertainty.level}</span>: ${sc.uncertainty.reason}</div></div>
        </div>

        <!-- Live Evaluation Breakdown -->
        <div class="detail-card" style="border-color: rgba(56, 189, 248, 0.4);">
          <div class="detail-heading">Stage 5 Audit Verification Breakdown</div>
          <div class="audit-grid">
            <div class="audit-item"><span>Schema Validation</span><strong class="${sValid === 'PASS' ? 'text-pass' : 'text-fail'}">${sValid}</strong></div>
            <div class="audit-item"><span>Provenance Audit</span><strong class="${pValid === 'PASS' ? 'text-pass' : 'text-fail'}">${pValid}</strong></div>
            <div class="audit-item"><span>Blind Spot Targeted</span><strong class="${bsValid === 'PASS' ? 'text-pass' : 'text-fail'}">${bsValid}</strong></div>
            <div class="audit-item"><span>Genomic Consistency</span><strong class="${gValid === 'PASS' ? 'text-pass' : 'text-fail'}">${gValid}</strong></div>
            <div class="audit-item"><span>Clinical Consistency</span><strong class="${clValid === 'PASS' ? 'text-pass' : 'text-fail'}">${clValid}</strong></div>
            <div class="audit-item"><span>Decision Stress Score</span><strong class="text-stress">${evalData.decision_stress_score?.overall_stress_score ?? 'N/A'} / 5.0</strong></div>
          </div>
          <div style="margin-top: 10px; font-size: 12px; color: var(--text-muted);">
            Realism Score: <strong class="text-realism">${evalData.realism_score?.overall_realism_score ?? 'N/A'} / 5.0</strong> | Identified Stress Dimensions: <em>${(evalData.resistance_stress_audit?.identified_dimensions || []).join(", ") || "None"}</em>
          </div>
        </div>

        <!-- Provenance -->
        <div class="detail-card">
          <div class="detail-heading">Provenance & Audit Trail</div>
          <div class="field-row"><div class="field-label">Generator:</div><div class="field-value">${sc.provenance.generation_method} (Seed: ${sc.provenance.random_seed})</div></div>
          <div class="field-row"><div class="field-label">Timestamp:</div><div class="field-value">${sc.provenance.generation_timestamp}</div></div>
          <div class="field-row"><div class="field-label">Sources:</div><div class="field-value">${(sc.provenance.reference_sources || []).join(", ")}</div></div>
        </div>
      `;
    }

    async function evaluateCurrentScenario() {
      if (!selectedScenarioId) return;
      await liveEvaluateRow(selectedScenarioId);
    }

    async function liveEvaluateRow(scenarioId) {
      try {
        const res = await fetch(`/stage5/scenarios/${scenarioId}/evaluate`, { method: "POST" });
        if (!res.ok) throw new Error("Evaluation request failed");
        await loadSummary();
        await loadScenarios();
        await loadHistory();
        selectScenario(scenarioId);
      } catch (err) {
        alert("Evaluation Error: " + err.message);
      }
    }

    async function runBatchEvaluation() {
      if (!confirm("Run batch evaluation across all synthetic scenarios?")) return;
      try {
        const res = await fetch("/stage5/scenarios/evaluate-all", { method: "POST" });
        if (!res.ok) throw new Error("Batch evaluation failed");
        const data = await res.json();
        alert(`Batch evaluation completed! Total: ${data.total_scenarios}, Passed: ${data.passed}, Review: ${data.review}, Failed: ${data.failed}`);
        await loadSummary();
        await loadScenarios();
        await loadHistory();
        if (selectedScenarioId) selectScenario(selectedScenarioId);
      } catch (err) {
        alert("Batch Evaluation Error: " + err.message);
      }
    }

    async function reloadScenarios() {
      try {
        await loadSummary();
        await loadScenarios();
        await loadHistory();
        alert("Scenarios and evaluation reports refreshed from disk.");
      } catch (err) {
        alert("Reload Error: " + err.message);
      }
    }

    async function loadHistory() {
      try {
        const res = await fetch("/stage5/evaluation/history");
        if (!res.ok) return;
        const records = await res.json();
        const tbody = document.getElementById("history-table-body");
        tbody.innerHTML = "";

        // Display latest first
        records.slice().reverse().forEach(r => {
          const tr = document.createElement("tr");
          let badgeClass = "badge-not-eval";
          if (r.evaluation_status === "PASS") badgeClass = "badge-pass";
          else if (r.evaluation_status === "REVIEW") badgeClass = "badge-review";
          else if (r.evaluation_status === "FAIL") badgeClass = "badge-fail";

          tr.innerHTML = `
            <td><span class="tag-mono">${r.run_id}</span></td>
            <td><strong style="color: #fff;">${r.scenario_id}</strong></td>
            <td><span class="tag-mono" style="color: var(--accent-amber);">${r.blind_spot_targeted}</span></td>
            <td><strong style="color: var(--accent-purple);">${typeof r.decision_stress_score === 'number' ? r.decision_stress_score.toFixed(2) : r.decision_stress_score}</strong></td>
            <td><strong style="color: var(--accent-blue);">${typeof r.realism_score === 'number' ? r.realism_score.toFixed(2) : r.realism_score}</strong></td>
            <td><span class="status-badge ${badgeClass}">${r.evaluation_status}</span></td>
            <td style="font-size: 11px; color: var(--text-muted);">${r.evaluation_timestamp}</td>
          `;
          tbody.appendChild(tr);
        });
      } catch (err) {
        console.error("Failed to load history", err);
      }
    }

    window.onload = init;
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the interactive Stage 5 testing dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


def main():
    """CLI entrypoint to launch the Stage 5 Dashboard server."""
    import uvicorn
    port = int(os.environ.get("STAGE5_PORT", "8080"))
    host = os.environ.get("STAGE5_HOST", "127.0.0.1")
    print(f"==================================================")
    print(f"Stage 5 Synthetic Scenario Testing Dashboard")
    print(f"Serving at: http://{host}:{port}/")
    print(f"API documentation: http://{host}:{port}/docs")
    print(f"==================================================")
    uvicorn.run("personalized_precision_oncology.stage5_genai.integration.src.dashboard_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
