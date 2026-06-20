# Unified Configuration and UI Refactoring Report

This document provides a comprehensive log of the enhancements, refactorings, and features added to the **Self-Organising RAG** system during this development session. All updates have been successfully verified, committed, and pushed to the remote branch `anantha`.

---

## end-to-end changes summary

```
┌────────────────────────────────────────────────────────────────────────┐
│                              DASHBOARD (app.py)                        │
│   ┌─────────────────────┐   ┌──────────────────────────────────────┐   │
│   │   Remove Eval Page  │   │  - System & Repair Settings Forms     │   │
│   │   (State preserved) │   │  - Auto-Refresh Loop (Every 10s)     │   │
│   │                     │   │  - Dynamic Rule Matrix (Live values) │   │
│   └─────────────────────┘   └───────────────────┬──────────────────┘   │
└─────────────────────────────────────────────────┼──────────────────────┘
                                                  │ (1) Write / Update
                                                  ▼
                                            ┌───────────┐
                                            │   .env    │
                                            └─────┬─────┘
                                                  │ (2) Auto-reload / Sync
                             ┌────────────────────┴────────────────────┐
                             ▼                                         ▼
                 ┌──────────────────────┐                  ┌───────────────────────┐
                 │  FastAPI Backend     │                  │  auto_worker.py       │
                 │  (uvicorn reloads)   │                  │  (polls env every 5s) │
                 └──────────────────────┘                  └───────────────────────┘
```

---

## 1. Remove Evaluation History Page

### Objective
Clean up the dashboard UI by removing the legacy "Eval History" page, preventing confusion and maintaining navigation focus.

### Changes Implemented
* **Sidebar Menu**: Modified the navigation radio buttons in [dashboard/app.py](file:///e:/CPP/anantha/dashboard/app.py#L665-L669) to exclude `"Eval History"`.
* **Rendering Logic**: Removed the entire execution block corresponding to `page == "Eval History"` (formerly lines 1652–1693) that queried `EvalSnapshot` logs and rendered the evaluation summary table and trend charts.

---

## 2. Unified Settings & Dynamic Threshold Configuration

### Objective
Unify all hardcoded parameters across various modules (daemons, endpoints, detectors, evaluation metrics, and cascades) into a single, cohesive configuration class `config.py` loaded from `.env`.

### Architecture Enhancements

#### Settings Schema Upgrades ([config.py](file:///e:/CPP/anantha/config.py))
The `Settings` class was expanded to define the following parameters along with their defaults:
* **Background Daemon (Auto-Worker)**:
  * `pending_min: int = 5` — Minimum flagged events required to trigger repair.
  * `pending_ratio: float = 0.30` — Ratio of total queries flagged required to trigger repair.
  * `poll_interval_seconds: int = 5` — Check interval of the database queue.
* **Low Recall Detectors**:
  * `score_low: float = 0.65` — Top-1 retrieval similarity floor.
  * `score_drop: float = 0.15` — Maximum adjacent rank score gap.
  * `coherence_ratio: float = 0.65` — Target coherence multiplier for semantic mismatch.
  * `evidence_match: float = 0.60` — Answer-to-evidence similarity floor.
* **Stage 2 Quality Metrics**:
  * `precision_relevance_threshold: float = 0.50` — Chunk-to-ground-truth sim to be relevant.
  * `hallucination_grounding_threshold: float = 0.55` — Claim-to-context sim to be grounded.
* **Repair & Promotion Limits**:
  * `score_cliff_threshold: float = 0.12` — Score cliff gap for dynamic K pruning.
  * `promotion_threshold: int = 5` — Strategy success counts needed for main pipeline promotion.

#### Dynamic Settings Loading
All backend modules were refactored to read configuration parameters dynamically at runtime:
* **Background Daemon ([auto_worker.py](file:///e:/CPP/anantha/auto_worker.py))**: Calls `settings.__init__(_env_file=".env")` inside the monitoring loop. If thresholds are modified from the dashboard UI, the running auto-worker daemon pulls the updated values on its next poll cycle (5s) dynamically.
* **Detection Engine ([detector/detectors.py](file:///e:/CPP/anantha/detector/detectors.py))**: Rewrote rule validators to pull `score_low`, `score_drop`, `coherence_ratio`, and `evidence_match` dynamically from settings.
* **Quality Metrics ([controllers/metrics.py](file:///e:/CPP/anantha/controllers/metrics.py))**: Tuned validation functions to retrieve precision relevance, context sufficiency, and hallucination grounding thresholds dynamically from configuration.
* **Diagnosis Engine ([detector/decision_engine.py](file:///e:/CPP/anantha/detector/decision_engine.py))**: Dynamically queries `settings.precision_threshold` and `settings.hallucination_threshold` in the root cause selector.
* **Cascade & Orchestrator ([repair/orchestrator.py](file:///e:/CPP/anantha/repair/orchestrator.py) & [repair/cascade.py](file:///e:/CPP/anantha/repair/cascade.py))**: Uses `settings.score_cliff_threshold` and `settings.promotion_threshold` dynamically inside K-selection and S1 promotion checks.

---

## 3. Dashboard Pipeline Config Page Extension

### Objective
Expose the unified system settings on the dashboard so they can be inspected and updated dynamically.

### UI Form Implementation ([dashboard/app.py](file:///e:/CPP/anantha/dashboard/app.py#L1703-L1827))
We integrated a **System & Repair Threshold Settings** form containing four main logical groups:
1. **Models & Provider Setup**: Text inputs for Primary LLM, Fallback LLM, Embedding Model, and Ollama Base URL.
2. **Auto-Worker Trigger Settings**: Numeric inputs for Min Pending Events, Poll Interval, and a slider for the Pending Ratio.
3. **Low Recall Trigger Thresholds**: Sliders for Top-1 Score floor, Adjacent gap cliff, Semantic coherence ratio, and Answer-evidence match.
4. **Quality Metrics & Repair Thresholds**: Sliders for Precision@K, Context Sufficiency, Hallucination grounding, and promotion successes limit.

### Sync & Persistence Logic
When the user submits the form:
1. The updates are saved back to the workspace `.env` file (updating standard `KEY=VALUE` pairs).
2. The `uvicorn` development server detects the `.env` change and automatically restarts the FastAPI backend.
3. The Streamlit process updates its in-memory `settings` object dynamically, rendering the new thresholds immediately.

---

## 4. Live Display & Diagnostics

### Objective
Ensure that user-facing lists and rule matrices update dynamically based on the active configuration.

### Changes Implemented
* **Overview Page Rule Matrix** ([dashboard/app.py](file:///e:/CPP/anantha/dashboard/app.py#L754-L787)): Refactored the "Detection Rule Matrix" HTML table into a Python f-string that dynamically prints `settings.score_low`, `settings.score_drop`, `settings.coherence_ratio`, and `settings.evidence_match`.
* **Query Diagnostics explanations** ([dashboard/app.py](file:///e:/CPP/anantha/dashboard/app.py#L1180-L1188)): Updated explanations for triggered events to dynamically insert active thresholds for better logging clarity.

---

## 5. State-Preserving Auto-Refresh

### Objective
Automatically refresh the dashboard metrics from the database without resetting the user's view or forcing navigation back to the default "Overview" page.

### Changes Implemented
* **Installed `streamlit-autorefresh`**: Proposed and ran `pip install streamlit-autorefresh` inside the virtual environment.
* **Updated [requirements.txt](file:///e:/CPP/anantha/requirements.txt)**: Appended `streamlit-autorefresh` to keep the dependency index updated.
* **Configured Autorefresh Loop**: Injected `st_autorefresh(interval=10000, key="auto_refresh_dashboard")` at the root of `dashboard/app.py`. Every 10 seconds, the frontend re-runs the Streamlit script, pulling updated databases records while keeping the sidebar selection and session state intact.

---

## 6. Git Provenance

The changes made across all 8 modules have been committed and pushed to the upstream branch:

```bash
# Staged changes
git add auto_worker.py config.py controllers/metrics.py dashboard/app.py detector/decision_engine.py detector/detectors.py repair/cascade.py repair/orchestrator.py requirements.txt

# Commits made
git commit -m "Unify system and repair thresholds under settings config and expose them in Pipeline Configuration page UI"
git commit -m "Integrate streamlit-autorefresh to refresh dashboard every 10 seconds while preserving page state"

# Pushed
git push origin anantha
```

---
*Created on: 2026-06-20*
