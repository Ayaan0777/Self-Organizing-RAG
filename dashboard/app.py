"""
Auto-RAG — Query Diagnostic Dashboard v3 (Cyberpunk Terminal Edition)
Run from project root:
  1. uvicorn main:app --reload          (start FastAPI backend)
  2. streamlit run dashboard/app.py     (start dashboard)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import requests
import streamlit as st
import pandas as pd
from db.session import get_session, init_db

from db.models import QueryLog, LowRecallEvent, EvalSnapshot, AdaptationLog, PipelineConfig

init_db()

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="AUTO-RAG // DIAGNOSTIC",
    page_icon="⬡",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════
#  CYBERPUNK TERMINAL CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Orbitron:wght@400;700;900&display=swap');

/* ── Root & Global ─────────────────────────────────────────────────── */
:root {
    --bg:        #050510;
    --surface:   #090920;
    --surface2:  #0d0d28;
    --border:    #1a1a45;
    --cyan:      #00f5ff;
    --cyan-dim:  #00b8c0;
    --green:     #00ff88;
    --amber:     #ffb800;
    --red:       #ff3366;
    --text:      #ffffff;
    --muted:     #4a5280;
    --font-mono: 'JetBrains Mono', monospace;
    --font-disp: 'Orbitron', monospace;
}

/* Global font & background */
html, body, [class*="css"], .stApp {
    font-family: var(--font-mono) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Scanline overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,245,255,0.015) 2px,
        rgba(0,245,255,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::before {
    content: '⬡ AUTO-RAG';
    display: block;
    font-family: var(--font-disp) !important;
    font-size: 0.85rem;
    font-weight: 900;
    color: var(--text);
    padding: 24px 20px 8px 20px;
    letter-spacing: 3px;
    text-shadow: 0 0 12px var(--text);
}
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    color: #ffffff !important;
    letter-spacing: 1px;
    padding: 6px 0 !important;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--cyan) !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-family: var(--font-disp) !important;
    font-size: 0.65rem !important;
    color: #a0a8cc !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Headings ───────────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: var(--font-disp) !important;
    letter-spacing: 2px !important;
}
h1 { color: var(--cyan) !important; text-shadow: 0 0 20px rgba(0,245,255,0.4) !important; font-size: 1.4rem !important; }
h2 { color: var(--text) !important; font-size: 1rem !important; }
h3 { color: var(--cyan-dim) !important; font-size: 0.9rem !important; }

/* ── Metric Cards ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--cyan) !important;
    border-radius: 4px !important;
    padding: 18px 20px !important;
    position: relative;
    transition: box-shadow 0.3s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 0 20px rgba(0,245,255,0.15), inset 0 0 20px rgba(0,245,255,0.03) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-disp) !important;
    font-size: 2rem !important;
    color: var(--cyan) !important;
    text-shadow: 0 0 15px rgba(0,245,255,0.5) !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    border-radius: 2px !important;
    padding: 8px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,245,255,0.1) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.3) !important;
    color: #fff !important;
}
.stButton > button[kind="primary"] {
    border-color: var(--green) !important;
    color: var(--green) !important;
}
.stButton > button[kind="primary"]:hover {
    background: rgba(0,255,136,0.1) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
}

/* ── Inputs & Selects ───────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
    caret-color: var(--cyan) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 10px rgba(0,245,255,0.2) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── File Uploader ──────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--cyan) !important;
    border-radius: 4px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(0,245,255,0.04) !important;
    box-shadow: 0 0 15px rgba(0,245,255,0.1) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-family: var(--font-mono) !important;
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
}

/* ── Dataframe / Tables ─────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
.dvn-scroller { background: var(--surface) !important; }

/* ── Info / Success / Warning / Error ───────────────────────────────── */
[data-testid="stAlert"] {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    border-radius: 2px !important;
    border-left-width: 3px !important;
    background: var(--surface2) !important;
}

/* ── Caption ────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: #ffffff !important;
    letter-spacing: 1px;
}

/* ── Charts ─────────────────────────────────────────────────────────── */
[data-testid="stArrowVegaLiteChart"] canvas,
[data-testid="stVegaLiteChart"] {
    filter: hue-rotate(160deg) saturate(1.5) brightness(1.1);
}

/* ── Scrollbar ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ── Custom Components ──────────────────────────────────────────────── */

/* Page header banner */
.page-header {
    border-left: 3px solid var(--cyan);
    padding: 10px 18px;
    margin-bottom: 24px;
    background: linear-gradient(90deg, rgba(0,245,255,0.06) 0%, transparent 100%);
}
.page-header .prefix {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--cyan);
    letter-spacing: 3px;
    text-transform: uppercase;
    opacity: 0.7;
}
.page-header h1 {
    font-family: var(--font-disp) !important;
    font-size: 1.3rem !important;
    margin: 4px 0 0 0;
    padding: 0;
}

/* Section header */
.sec-header {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--cyan);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 28px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}
.sec-header::before { content: '▸ '; opacity: 0.6; }

/* Status dot */
.status-dot {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    letter-spacing: 1px;
}
.dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-dot 2s infinite;
}
.dot-green  { background: var(--green);  box-shadow: 0 0 8px var(--green); }
.dot-red    { background: var(--red);    box-shadow: 0 0 8px var(--red); animation: none; }
.dot-amber  { background: var(--amber);  box-shadow: 0 0 8px var(--amber); }
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* Chunk cards */
.chunk-card {
    background: var(--surface2);
    border-left: 3px solid;
    border-radius: 2px;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    line-height: 1.6;
    position: relative;
}
.chunk-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, currentColor, transparent);
    opacity: 0.3;
}
.chunk-rank {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 8px;
    text-transform: uppercase;
    opacity: 0.9;
}
.chunk-text { color: #8899cc; font-size: 0.8rem; }
.score-high { border-color: var(--green); color: var(--green); }
.score-mid  { border-color: var(--amber); color: var(--amber); }
.score-low  { border-color: var(--red);   color: var(--red);   }

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 1px 10px;
    border-radius: 2px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-left: 10px;
    border: 1px solid currentColor;
}
.badge-high { color: var(--green); background: rgba(0,255,136,0.1); }
.badge-mid  { color: var(--amber); background: rgba(255,184,0,0.1); }
.badge-low  { color: var(--red);   background: rgba(255,51,102,0.1); }

/* Answer card */
.answer-card {
    background: var(--surface2);
    border-left: 3px solid var(--cyan);
    border-radius: 2px;
    padding: 20px 24px;
    margin: 12px 0;
    position: relative;
}
.answer-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--cyan), transparent);
    opacity: 0.3;
}
.answer-label {
    font-size: 0.65rem;
    color: var(--cyan);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    opacity: 0.8;
}
.answer-label::before { content: '▸ '; }
.answer-text  { color: var(--text); font-size: 0.88rem; line-height: 1.8; }

/* Severity badges */
.sev-high   { color: var(--red);   font-weight: 700; letter-spacing: 1px; }
.sev-medium { color: var(--amber); font-weight: 600; letter-spacing: 1px; }
.sev-low    { color: var(--green); letter-spacing: 1px; }

/* Detection rules table */
.rules-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.78rem;
}
.rules-table th {
    text-align: left;
    color: var(--cyan);
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
}
.rules-table td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(26,26,69,0.5);
    color: #8899cc;
    vertical-align: top;
}
.rules-table tr:hover td { background: rgba(0,245,255,0.03); color: var(--text); }
.rules-table code {
    background: rgba(0,245,255,0.08);
    color: var(--cyan);
    padding: 2px 7px;
    border-radius: 2px;
    font-size: 0.75rem;
    border: 1px solid rgba(0,245,255,0.2);
}
.rules-table .thresh {
    color: var(--amber);
    font-weight: 700;
}

/* Flagging reason card */
.flag-reason {
    background: rgba(255,51,102,0.06);
    border-left: 3px solid var(--red);
    border-radius: 2px;
    padding: 8px 14px;
    margin: 6px 0;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: #cc8899;
}

/* Terminal-style section divider */
.term-div {
    border: none;
    border-top: 1px solid var(--border);
    margin: 28px 0 20px 0;
    position: relative;
}
.term-div::after {
    content: '────';
    position: absolute;
    top: -9px;
    left: 0;
    font-size: 0.6rem;
    color: var(--border);
    background: var(--bg);
    padding-right: 6px;
    letter-spacing: -2px;
}
/* ── Data Tables (custom, fully themed) ─────────────────────────── */
.rag-table-wrap {
    width: 100%;
    overflow-x: auto;
    margin-bottom: 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
}
.rag-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.78rem;
}
.rag-table thead tr {
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
}
.rag-table th {
    text-align: left;
    color: var(--cyan);
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 14px;
    white-space: nowrap;
}
.rag-table tbody tr {
    border-bottom: 1px solid rgba(26,26,69,0.6);
    transition: background 0.15s;
}
.rag-table tbody tr:nth-child(even) { background: rgba(13,13,40,0.5); }
.rag-table tbody tr:hover { background: rgba(0,245,255,0.04); }
.rag-table td {
    padding: 9px 14px;
    color: #8899cc;
    vertical-align: middle;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════
def check_backend():
    try:
        resp = requests.get(f"{API_BASE}/logs?limit=1", timeout=8)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout, requests.ReadTimeout):
        return False

def page_header(prefix, title):
    st.markdown(f"""
    <div class="page-header">
        <div class="prefix">{prefix}</div>
        <h1>{title}</h1>
    </div>
    """, unsafe_allow_html=True)

def sec_header(title):
    st.markdown(f'<div class="sec-header">{title}</div>', unsafe_allow_html=True)

def term_div():
    st.markdown('<hr class="term-div">', unsafe_allow_html=True)

def render_table(df):
    """Render a DataFrame as a fully themed HTML table."""
    cols = df.columns.tolist()
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in cols:
            val = str(row[col])
            # Colour-code status / severity / resolved cells
            if val in ("⚠ FLAGGED",):
                val = f'<span style="color:var(--amber);font-weight:700;">{val}</span>'
            elif val in ("✓ OK",):
                val = f'<span style="color:var(--green);">{val}</span>'
            elif val == "HIGH":
                val = f'<span style="color:var(--red);font-weight:700;">HIGH</span>'
            elif val == "MEDIUM":
                val = f'<span style="color:var(--amber);font-weight:600;">MEDIUM</span>'
            elif val == "LOW":
                val = f'<span style="color:var(--green);">LOW</span>'
            elif val in ("✓ YES",):
                val = f'<span style="color:var(--green);">{val}</span>'
            elif val in ("✗ NO",):
                val = f'<span style="color:var(--muted);">{val}</span>'
            cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"
    html = f"""
    <div class="rag-table-wrap">
        <table class="rag-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def backend_status():
    up = check_backend()
    if up:
        st.markdown('<div class="status-dot"><span class="dot dot-green"></span> BACKEND ONLINE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-dot"><span class="dot dot-red"></span> BACKEND OFFLINE — run: <code>uvicorn main:app --reload</code></div>', unsafe_allow_html=True)
    return up

def chunk_card(i, chunk, score):
    if score >= 0.7:
        card_cls, badge_cls, label = "score-high", "badge-high", "HIGH"
    elif score >= 0.45:
        card_cls, badge_cls, label = "score-mid",  "badge-mid",  "MED"
    else:
        card_cls, badge_cls, label = "score-low",  "badge-low",  "LOW"
    display_text = chunk[:500] + ("…" if len(chunk) > 500 else "")
    st.markdown(f"""
    <div class="chunk-card {card_cls}">
        <div class="chunk-rank">
            CHUNK_{i+1:02d} &nbsp;·&nbsp; SIM: {score:.4f}
            <span class="score-badge {badge_cls}">{label}</span>
        </div>
        <div class="chunk-text">{display_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  TITLE & NAV
# ══════════════════════════════════════════════════════════════════════════
session = get_session()

page = st.sidebar.radio(
    "NAVIGATE",
    ["Overview", "Ingest Document", "Ask Query", "Query Diagnostics",
     "Flagged Events", "Eval History", "Pipeline Config", "Adaptation Log"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<span style='font-size:0.65rem;color:#2a2a60;letter-spacing:2px;'>v3 // TERMINAL EDITION</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    page_header("SYS // MONITOR", "SYSTEM OVERVIEW")

    total    = session.query(QueryLog).count()
    flagged  = session.query(QueryLog).filter(QueryLog.flagged == True).count()
    healthy  = total - flagged
    events   = session.query(LowRecallEvent).count()
    resolved = session.query(LowRecallEvent).filter(LowRecallEvent.resolved == True).count()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL QUERIES",  total)
    c2.metric("HEALTHY",        healthy)
    c3.metric("FLAGGED",        flagged)
    c4.metric("EVENTS",         events)

    flag_rate = round(flagged / total * 100, 1) if total else 0
    res_rate  = round(resolved / events * 100, 1) if events else 0
    st.caption(f"FLAG RATE: {flag_rate}%  ·  RESOLUTION RATE: {res_rate}%")

    # ── Stage 2 Metric Gauges ──
    term_div()
    sec_header("STAGE 2 — MEASURE METRICS (LATEST EVAL)")
    latest_eval = session.query(EvalSnapshot).order_by(EvalSnapshot.timestamp.desc()).first()
    if latest_eval:
        m1, m2, m3, m4 = st.columns(4)
        rp = latest_eval.retrieval_precision
        cs = latest_eval.context_sufficiency
        hr = latest_eval.hallucination_rate
        m1.metric("RETR. PRECISION", f"{rp:.2%}" if rp is not None else "—")
        m2.metric("CTX SUFFICIENCY", f"{cs:.2%}" if cs is not None else "—")
        m3.metric("HALLUCINATION",   f"{hr:.2%}" if hr is not None else "—")

        # Pipeline config status
        active_cfg = session.query(PipelineConfig).filter(
            PipelineConfig.active == True
        ).order_by(PipelineConfig.created_at.desc()).first()
        if active_cfg:
            m4.metric("CHUNK SIZE", f"{active_cfg.chunk_size} / {active_cfg.chunk_overlap}")
        else:
            m4.metric("CHUNK SIZE", "250 / 80 (default)")

        # Adaptation stats
        total_adaptations = session.query(AdaptationLog).count()
        rollbacks = session.query(AdaptationLog).filter(AdaptationLog.rolled_back == True).count()
        improvements = session.query(AdaptationLog).filter(AdaptationLog.outcome == "IMPROVED").count()
        st.caption(
            f"ADAPTATIONS: {total_adaptations}  ·  "
            f"IMPROVED: {improvements}  ·  "
            f"ROLLED BACK: {rollbacks}"
        )
    else:
        st.info("▸ NO EVALUATION DATA — run an evaluation to see Stage 2 metrics.")

    term_div()

    all_rows = session.query(QueryLog).order_by(QueryLog.timestamp.asc()).limit(500).all()
    if all_rows:
        trend_data = []
        for r in all_rows:
            scores = json.loads(r.top_k_scores or "[]")
            if scores:
                trend_data.append({
                    "timestamp": r.timestamp,
                    "top1_score": scores[0],
                    "flagged": r.flagged,
                })
        if trend_data:
            

            col1, col2 = st.columns(2)
            with col1:
                sec_header("SCORE DISTRIBUTION")
                scores_all = [d["top1_score"] for d in trend_data]
                hist_series = pd.cut(pd.Series(scores_all), bins=10).value_counts().sort_index()
                hist_series.index = hist_series.index.astype(str)
                st.bar_chart(hist_series, height=200)
            with col2:
                sec_header("FLAGGED vs HEALTHY")
                flag_counts = pd.Series({"HEALTHY": healthy, "FLAGGED": flagged})
                st.bar_chart(flag_counts, height=200)
    else:
        st.info("▸ NO QUERIES LOGGED — send queries via /api/v1/query to begin tracking.")

    term_div()
    sec_header("DETECTION RULE MATRIX")
    st.markdown("""
    <table class="rules-table">
        <tr>
            <th>RULE</th>
            <th>TRIGGER CONDITION</th>
            <th>THRESHOLD</th>
        </tr>
        <tr>
            <td><code>low_top_score</code></td>
            <td>Best chunk similarity below floor</td>
            <td class="thresh">0.45</td>
        </tr>
        <tr>
            <td><code>score_drop</code></td>
            <td>Gap between rank-1 and rank-k too large</td>
            <td class="thresh">0.30</td>
        </tr>
        <tr>
            <td><code>llm_uncertainty</code></td>
            <td>Response contains hedging language</td>
            <td class="thresh">keyword</td>
        </tr>
        <tr>
            <td><code>semantic_mismatch</code></td>
            <td>Chunks are semantically fragmented</td>
            <td class="thresh">0.55</td>
        </tr>
        <tr>
            <td><code>evidence_mismatch</code></td>
            <td>LLM answer diverges from retrieved context</td>
            <td class="thresh">0.50</td>
        </tr>
        <tr>
            <td><code>user_frustration</code></td>
            <td>Similar query re-asked within 5 min</td>
            <td class="thresh">0.85</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  INGEST DOCUMENT
# ══════════════════════════════════════════════════════════════════════════
elif page == "Ingest Document":
    page_header("PIPELINE // INPUT", "DOCUMENT INGEST")
    backend_up = backend_status()

    term_div()

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "TARGET FILE",
            type=["pdf", "docx", "txt"],
            help="Supported: PDF · DOCX · TXT",
        )
    with col2:
        namespace = st.text_input(
            "NAMESPACE (optional)",
            value="",
            help="Leave blank for default namespace from .env",
        )

    if uploaded_file is not None:
        st.markdown(
            f'<div class="status-dot" style="margin:12px 0"><span class="dot dot-amber"></span>'
            f' {uploaded_file.name} &nbsp;·&nbsp; {uploaded_file.size / 1024:.1f} KB</div>',
            unsafe_allow_html=True,
        )

        if st.button("▸ INGEST DOCUMENT", type="primary", disabled=not backend_up):
            with st.spinner("INGESTING — please wait..."):
                try:
                    files  = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
                    params = {}
                    if namespace.strip():
                        params["namespace"] = namespace.strip()

                    resp = requests.post(f"{API_BASE}/ingest", files=files, params=params, timeout=300)

                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(f"▸ INGEST COMPLETE — {result.get('message', 'Document ingested.')}")
                        st.json(result)
                    else:
                        st.error(f"▸ INGEST FAILED — HTTP {resp.status_code}")
                        st.code(resp.text)

                except requests.ConnectionError:
                    st.error("▸ CONNECTION ERROR — backend unreachable")
                except requests.Timeout:
                    st.error("▸ TIMEOUT — file may be too large")
                except Exception as e:
                    st.error(f"▸ ERROR — {e}")


# ══════════════════════════════════════════════════════════════════════════
#  ASK QUERY
# ══════════════════════════════════════════════════════════════════════════
elif page == "Ask Query":
    page_header("PIPELINE // QUERY", "QUERY INTERFACE")
    backend_up = backend_status()

    term_div()

    query_text = st.text_area(
        "INPUT QUERY",
        height=100,
        placeholder="// enter query string...",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        namespace = st.text_input("NAMESPACE", value="", key="query_ns")

    submit_query = st.button("▸ EXECUTE QUERY", type="primary", disabled=not backend_up or not query_text.strip())

    if submit_query and query_text.strip():
        with st.spinner("RETRIEVING CONTEXT · GENERATING ANSWER..."):
            try:
                payload = {"query": query_text.strip()}
                if namespace.strip():
                    payload["namespace"] = namespace.strip()

                t0   = time.time()
                resp = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
                elapsed = time.time() - t0

                if resp.status_code == 200:
                    result   = resp.json()
                    answer   = result.get("answer", "No answer returned.")
                    contexts = result.get("retrieved_contexts", [])
                    scores   = result.get("scores", [])
                    log_id   = result.get("log_id", "—")

                    term_div()
                    sec_header("GENERATED ANSWER")
                    st.markdown(f"""
                    <div class="answer-card">
                        <div class="answer-label">RAG OUTPUT</div>
                        <div class="answer-text">{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    top_score = scores[0] if scores else 0
                    m1.metric("TOP-1 SCORE",    f"{top_score:.4f}")
                    m2.metric("LATENCY",         f"{elapsed:.2f}s")
                    m3.metric("LOG ID",           str(log_id))

                    if contexts and scores:
                        term_div()
                        sec_header("RETRIEVED CHUNKS")
                        for i, (chunk, score) in enumerate(zip(contexts, scores)):
                            chunk_card(i, chunk, score)

                    if scores:
                        term_div()
                        sec_header("SCORE DISTRIBUTION")
                        chart_df = pd.DataFrame({
                            "Chunk": [f"#{i+1}" for i in range(len(scores))],
                            "Relevance Score": scores,
                        })
                        st.bar_chart(chart_df.set_index("Chunk"), height=220)

                else:
                    st.error(f"▸ QUERY FAILED — HTTP {resp.status_code}")
                    st.code(resp.text)

            except requests.ConnectionError:
                st.error("▸ CONNECTION ERROR — backend unreachable")
            except requests.Timeout:
                st.error("▸ TIMEOUT — LLM response too slow")
            except Exception as e:
                st.error(f"▸ ERROR — {e}")


# ══════════════════════════════════════════════════════════════════════════
#  QUERY DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Query Diagnostics":
    page_header("ANALYSIS // LOGS", "QUERY DIAGNOSTICS")

    rows = session.query(QueryLog).order_by(QueryLog.timestamp.desc()).limit(100).all()

    if not rows:
        st.info("▸ NO QUERIES LOGGED — use Ask Query page to populate this view.")
    else:
        sec_header("QUERY LOG SUMMARY")
        summary_data = []
        for r in rows:
            cqs  = getattr(r, "ctx_q_sim", None)
            asim = getattr(r, "answer_sem_sim", None)
            summary_data.append({
                "ID":            r.id,
                "Query":         r.query[:90],
                "Ctx↔Q Sim":    f"{cqs:.4f}"  if cqs  is not None else "—",
                "Answer↔GT":    f"{asim:.4f}" if asim is not None else "—",
                "Status":        "⚠ FLAGGED" if r.flagged else "✓ OK",
                "Latency (s)":  f"{r.latency_ms / 1000:.2f}",
                "Time":          str(r.timestamp)[:19],
            })
        render_table(pd.DataFrame(summary_data))

        term_div()
        sec_header("QUERY INSPECTOR")

        query_options   = {f"#{r.id} — {r.query[:70]}": r.id for r in rows}
        selected_label  = st.selectbox("SELECT QUERY", list(query_options.keys()))
        selected_id     = query_options[selected_label]
        selected        = session.query(QueryLog).filter(QueryLog.id == selected_id).first()

        if selected:
            scores = json.loads(selected.top_k_scores or "[]")
            chunks = json.loads(selected.retrieved_chunks or "[]") if selected.retrieved_chunks else []

            cqs    = getattr(selected, "ctx_q_sim", None)
            asim   = getattr(selected, "answer_sem_sim", None)
            has_gt = asim is not None

            if has_gt:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("CTX↔QUERY SIM",  f"{cqs:.4f}" if cqs else "N/A")
                col2.metric("ANSWER↔GT SIM",  f"{asim:.4f}")
                col3.metric("LATENCY",          f"{selected.latency_ms / 1000:.2f}s")
                col4.metric("STATUS",           "⚠ FLAGGED" if selected.flagged else "✓ OK")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("CTX↔QUERY SIM",  f"{cqs:.4f}" if cqs else "N/A")
                col2.metric("LATENCY",          f"{selected.latency_ms / 1000:.2f}s")
                col3.metric("STATUS",           "⚠ FLAGGED" if selected.flagged else "✓ OK")
                st.caption("NO GROUND TRUTH — answer similarity not computed")

            # ── Stage 2 Metrics for this query ──
            rp = getattr(selected, "retrieval_precision", None)
            cs = getattr(selected, "context_sufficiency", None)
            hr = getattr(selected, "hallucination_rate", None)
            qc = getattr(selected, "question_category", None)
            if any(v is not None for v in [rp, cs, hr, qc]):
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("RETR. PRECISION",  f"{rp:.2%}" if rp is not None else "—")
                s2.metric("CTX SUFFICIENT",   "✓ YES" if cs else "✗ NO" if cs is not None else "—")
                s3.metric("HALLUCINATION",    f"{hr:.2%}" if hr is not None else "—")
                s4.metric("Q CATEGORY",       (qc or "—").upper())

            term_div()
            sec_header("LLM RESPONSE")
            ans_text = selected.llm_response or "_no response recorded_"
            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-text">{ans_text}</div>
            </div>
            """, unsafe_allow_html=True)

            if chunks and scores:
                term_div()
                sec_header("RETRIEVED CHUNKS & SCORES")
                for i, (chunk, score) in enumerate(zip(chunks, scores)):
                    chunk_card(i, chunk, score)
            elif scores:
                st.caption("Chunk text unavailable for this query (pre-upgrade log)")
                for i, score in enumerate(scores):
                    badge = "🟢" if score >= 0.7 else ("🟡" if score >= 0.45 else "🔴")
                    st.markdown(f"{badge} `CHUNK_{i+1:02d}` — `{score:.4f}`")
            else:
                st.warning("▸ NO SCORES RECORDED FOR THIS QUERY")

            if scores:
                term_div()
                sec_header("PER-QUERY SCORE DISTRIBUTION")
                chart_df = pd.DataFrame({
                    "Chunk": [f"#{i+1}" for i in range(len(scores))],
                    "Relevance Score": scores,
                })
                st.bar_chart(chart_df.set_index("Chunk"), height=220)

            if selected.flagged:
                event = session.query(LowRecallEvent).filter(
                    LowRecallEvent.query_log_id == selected.id
                ).first()
                if event:
                    detectors = json.loads(event.triggered_detectors or "[]")
                    term_div()
                    sec_header("FLAGGING ANALYSIS")
                    explanations = {
                        "low_top_score":    "LOW_TOP_SCORE — best retrieved chunk scored below 0.45",
                        "score_drop":       "SCORE_DROP — large gap between rank-1 and rank-K scores",
                        "llm_uncertainty":  "LLM_UNCERTAINTY — response contains hedging language",
                        "semantic_mismatch":"SEMANTIC_MISMATCH — retrieved chunks cover different topics",
                        "evidence_mismatch":"EVIDENCE_MISMATCH — LLM answer diverges from retrieved context",
                        "user_frustration": "USER_FRUSTRATION — similar query re-asked within 5 minutes",
                    }
                    for d in detectors:
                        st.markdown(
                            f'<div class="flag-reason">▸ {explanations.get(d, d)}</div>',
                            unsafe_allow_html=True,
                        )
                    sev_map = {
                        "HIGH":   '<span class="sev-high">HIGH</span>',
                        "MEDIUM": '<span class="sev-medium">MEDIUM</span>',
                        "LOW":    '<span class="sev-low">LOW</span>',
                    }
                    st.markdown(
                        f'<div style="font-family:var(--font-mono);font-size:0.8rem;margin-top:12px;color:#4a5280;letter-spacing:1px;">SEVERITY: {sev_map.get(event.severity, event.severity)}</div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════
#  FLAGGED EVENTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Flagged Events":
    page_header("ANALYSIS // EVENTS", "FLAGGED EVENTS")

    col1, col2 = st.columns(2)
    with col1:
        sev_filter = st.selectbox("FILTER BY SEVERITY", ["ALL", "HIGH", "MEDIUM", "LOW"])
    with col2:
        res_filter = st.selectbox("FILTER BY RESOLVED", ["ALL", "RESOLVED", "UNRESOLVED"])

    q = session.query(LowRecallEvent).order_by(LowRecallEvent.timestamp.desc())
    if sev_filter != "ALL":
        q = q.filter(LowRecallEvent.severity == sev_filter)
    if res_filter == "RESOLVED":
        q = q.filter(LowRecallEvent.resolved == True)
    elif res_filter == "UNRESOLVED":
        q = q.filter(LowRecallEvent.resolved == False)
    rows = q.limit(200).all()

    if not rows:
        st.success("▸ NO FLAGGED EVENTS — pipeline is operating nominally.")
    else:
        data = []
        for r in rows:
            detectors = json.loads(r.triggered_detectors or "[]")
            log       = session.query(QueryLog).filter(QueryLog.id == r.query_log_id).first()
            data.append({
                "ID":        r.id,
                "Query":     (log.query[:60] if log else "—"),
                "Severity":  r.severity,
                "Detectors": ", ".join(detectors),
                "Resolved":  "✓ YES" if r.resolved else "✗ NO",
                "Time":      str(r.timestamp)[:19],
            })
        render_table(pd.DataFrame(data))

        term_div()
        sec_header("REPAIR REPORT")
        event_options = {f"EVENT #{r.id} - QUERY LOG #{r.query_log_id}": r.id for r in rows}
        selected_event_label = st.selectbox("SELECT EVENT", list(event_options.keys()))
        selected_event_id = event_options[selected_event_label]

        try:
            report_resp = requests.get(f"{API_BASE}/repair-report/{selected_event_id}", timeout=30)
            if report_resp.status_code == 200:
                report = report_resp.json()

                # ── Row 1: Strategy & Scores ──
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("STRATEGY USED", report.get("strategy_used") or "N/A")
                col2.metric("SCORE BEFORE", report.get("score_before") if report.get("score_before") is not None else "N/A")
                col3.metric("SCORE AFTER", report.get("score_after") if report.get("score_after") is not None else "N/A")
                col4.metric("RESOLVED STATUS", "YES" if report.get("resolved") else "NO")

                # ── Row 2: Diagnosis Reason ──
                root_cause = report.get("root_cause")
                reasoning = report.get("reasoning")
                q_category = report.get("question_category")
                if root_cause or reasoning:
                    term_div()
                    sec_header("DIAGNOSIS — REASON FOR RESOLUTION")
                    d1, d2 = st.columns([1, 2])
                    with d1:
                        st.metric("ROOT CAUSE", (root_cause or "N/A").upper().replace("_", " "))
                        st.metric("QUESTION TYPE", (q_category or "N/A").upper().replace("_", " "))
                    with d2:
                        st.markdown(f"""
                        <div class="answer-card" style="border-left-color: var(--amber);">
                            <div class="answer-label" style="color: var(--amber);">REASONING</div>
                            <div class="answer-text" style="font-size: 0.82rem;">{reasoning or 'No diagnosis available.'}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Row 3: Enhanced Metrics (Precision, Recall, Accuracy) ──
                has_metrics = any(report.get(k) is not None for k in [
                    "precision_before", "recall_before", "accuracy_before"
                ])
                if has_metrics:
                    term_div()
                    sec_header("IMPROVEMENT SCORES")

                    def fmt_metric(val):
                        return f"{val:.2%}" if val is not None else "—"

                    def fmt_delta(before, after):
                        if before is not None and after is not None:
                            delta = after - before
                            sign = "+" if delta >= 0 else ""
                            return f"{sign}{delta:.2%}"
                        return None

                    m1, m2, m3 = st.columns(3)
                    m1.metric(
                        "CONTEXT PRECISION",
                        fmt_metric(report.get("precision_after")),
                        delta=fmt_delta(report.get("precision_before"), report.get("precision_after")),
                    )
                    m2.metric(
                        "RECALL",
                        fmt_metric(report.get("recall_after")),
                        delta=fmt_delta(report.get("recall_before"), report.get("recall_after")),
                    )
                    m3.metric(
                        "ANSWER ACCURACY",
                        fmt_metric(report.get("accuracy_after")),
                        delta=fmt_delta(report.get("accuracy_before"), report.get("accuracy_after")),
                    )
                    st.caption(
                        f"BEFORE → AFTER  ·  "
                        f"Precision: {fmt_metric(report.get('precision_before'))} → {fmt_metric(report.get('precision_after'))}  ·  "
                        f"Recall: {fmt_metric(report.get('recall_before'))} → {fmt_metric(report.get('recall_after'))}  ·  "
                        f"Accuracy: {fmt_metric(report.get('accuracy_before'))} → {fmt_metric(report.get('accuracy_after'))}"
                    )

                # ── Row 4: Answers ──
                term_div()
                st.text_area("ORIGINAL ANSWER", value=report.get("original_answer") or "", height=180, disabled=True)
                st.text_area("RESOLVED ANSWER", value=report.get("resolved_answer") or "", height=180, disabled=True)
            elif report_resp.status_code == 404:
                st.info("NO REPAIR REPORT FOUND FOR THIS EVENT")
            else:
                st.error(f"REPAIR REPORT FAILED - HTTP {report_resp.status_code}")
                st.code(report_resp.text)
        except requests.ConnectionError:
            st.error("CONNECTION ERROR - backend unreachable")
        except requests.Timeout:
            st.error("TIMEOUT - repair report request took too long")
        except Exception as e:
            st.error(f"ERROR - {e}")
elif page == "Eval History":
    page_header("ANALYSIS // EVAL", "EVALUATION HISTORY")

    rows = session.query(EvalSnapshot).order_by(EvalSnapshot.timestamp.desc()).limit(100).all()

    if not rows:
        st.info("▸ NO EVAL SNAPSHOTS — run: python run_evaluation.py")
    else:
        data = [{
            "Namespace":         r.namespace,
            "LLM":               r.llm,
            "Embeddings":        r.embeddings,
            "ROUGE-L":           r.rouge_l,
            "Semantic Sim":      r.sem_sim,
            "Ctx↔Query":        r.ctx_q_sim,
            "Ctx↔GT":           r.ctx_gt_sim,
            "Ret. Precision":    f"{r.retrieval_precision:.2%}" if r.retrieval_precision is not None else "—",
            "Ctx Sufficiency":   f"{r.context_sufficiency:.2%}" if r.context_sufficiency is not None else "—",
            "Hallucination":     f"{r.hallucination_rate:.2%}" if r.hallucination_rate is not None else "—",
            "Time":              str(r.timestamp)[:19],
        } for r in rows]
        render_table(pd.DataFrame(data))

        # Trend chart for new metrics across evaluation runs
        if len(rows) > 1:
            term_div()
            sec_header("METRIC TRENDS ACROSS EVALUATIONS")
            trend = []
            for r in reversed(rows):
                entry = {"Run": str(r.timestamp)[:16]}
                if r.retrieval_precision is not None:
                    entry["Retrieval Precision"] = r.retrieval_precision
                if r.hallucination_rate is not None:
                    entry["Hallucination Rate"] = r.hallucination_rate
                if r.sem_sim is not None:
                    entry["Semantic Similarity"] = r.sem_sim
                trend.append(entry)
            if trend:
                trend_df = pd.DataFrame(trend).set_index("Run")
                st.line_chart(trend_df, height=280)


# ══════════════════════════════════════════════════════════════════════════
#  PIPELINE CONFIG
# ══════════════════════════════════════════════════════════════════════════
elif page == "Pipeline Config":
    page_header("CONFIG // PIPELINE", "PIPELINE CONFIGURATION")

    # ── Current Active Config ──
    sec_header("CURRENT ACTIVE CONFIGURATION")
    active_cfg = session.query(PipelineConfig).filter(
        PipelineConfig.active == True
    ).order_by(PipelineConfig.created_at.desc()).first()

    if active_cfg:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CHUNK SIZE", active_cfg.chunk_size)
        c2.metric("CHUNK OVERLAP", active_cfg.chunk_overlap)
        c3.metric("STRATEGY", active_cfg.chunk_strategy.upper())
        c4.metric("NAMESPACE", active_cfg.namespace or "default")
        st.caption(f"Config ID: {active_cfg.id}  ·  Created: {str(active_cfg.created_at)[:19]}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("CHUNK SIZE", 250)
        c2.metric("CHUNK OVERLAP", 80)
        c3.metric("STRATEGY", "SEMANTIC")
        st.caption("Using system defaults — no custom config saved yet.")

    # ── Manual Override ──
    term_div()
    sec_header("MANUAL OVERRIDE")
    backend_up = backend_status()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        new_size = st.number_input("CHUNK SIZE", min_value=100, max_value=2000, value=250, step=50)
    with col2:
        new_overlap = st.number_input("OVERLAP", min_value=0, max_value=500, value=80, step=10)
    with col3:
        new_strategy = st.selectbox("STRATEGY", ["semantic", "llm", "entropy"])
    with col4:
        new_ns = st.text_input("NAMESPACE", value="", key="cfg_ns")

    if st.button("▸ APPLY CONFIGURATION", type="primary", disabled=not backend_up):
        try:
            params = {
                "chunk_size": new_size,
                "chunk_overlap": new_overlap,
                "chunk_strategy": new_strategy,
            }
            if new_ns.strip():
                params["namespace"] = new_ns.strip()
            resp = requests.post(f"{API_BASE}/pipeline-config", params=params, timeout=10)
            if resp.status_code == 200:
                st.success(f"▸ CONFIG UPDATED — size={new_size}, overlap={new_overlap}, strategy={new_strategy}")
                st.json(resp.json())
            else:
                st.error(f"▸ FAILED — HTTP {resp.status_code}")
        except Exception as e:
            st.error(f"▸ ERROR — {e}")

    # ── Config History ──
    term_div()
    sec_header("CONFIGURATION HISTORY")
    configs = session.query(PipelineConfig).order_by(PipelineConfig.created_at.desc()).limit(20).all()
    if configs:
        cfg_data = [{
            "ID":        c.id,
            "Size":      c.chunk_size,
            "Overlap":   c.chunk_overlap,
            "Strategy":  c.chunk_strategy,
            "Active":    "✓ YES" if c.active else "✗ NO",
            "Namespace": c.namespace or "—",
            "Created":   str(c.created_at)[:19],
        } for c in configs]
        render_table(pd.DataFrame(cfg_data))
    else:
        st.info("▸ NO CONFIG HISTORY — system is using defaults.")


# ══════════════════════════════════════════════════════════════════════════
#  ADAPTATION LOG (PROVENANCE TRAIL)
# ══════════════════════════════════════════════════════════════════════════
elif page == "Adaptation Log":
    page_header("AUDIT // PROVENANCE", "ADAPTATION LOG")

    # ── Summary Stats ──
    total_a = session.query(AdaptationLog).count()
    improved_a = session.query(AdaptationLog).filter(AdaptationLog.outcome == "IMPROVED").count()
    degraded_a = session.query(AdaptationLog).filter(AdaptationLog.outcome == "DEGRADED").count()
    nochange_a = session.query(AdaptationLog).filter(AdaptationLog.outcome == "NO_CHANGE").count()
    rollback_a = session.query(AdaptationLog).filter(AdaptationLog.rolled_back == True).count()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("TOTAL", total_a)
    c2.metric("IMPROVED", improved_a)
    c3.metric("DEGRADED", degraded_a)
    c4.metric("NO CHANGE", nochange_a)
    c5.metric("ROLLED BACK", rollback_a)

    if total_a:
        success_rate = round(improved_a / total_a * 100, 1)
        st.caption(f"SUCCESS RATE: {success_rate}%  ·  ROLLBACK RATE: {round(rollback_a / total_a * 100, 1)}%")

    # ── Timeline Table ──
    term_div()
    sec_header("ADAPTATION TIMELINE")
    rows = session.query(AdaptationLog).order_by(AdaptationLog.created_at.desc()).limit(50).all()

    if not rows:
        st.info("▸ NO ADAPTATIONS RECORDED — the self-healing loop has not triggered yet.")
    else:
        data = []
        for r in rows:
            # Parse diagnosis JSON for display
            try:
                diag = json.loads(r.diagnosis or "{}")
            except Exception:
                diag = {}
            try:
                obs = json.loads(r.observation or "{}")
            except Exception:
                obs = {}

            outcome_display = r.outcome or "—"
            if outcome_display == "IMPROVED":
                outcome_display = "✓ IMPROVED"
            elif outcome_display == "DEGRADED":
                outcome_display = "✗ DEGRADED"

            data.append({
                "Event":     r.event_id,
                "Strategy":  r.strategy_selected or "—",
                "Root Cause": diag.get("root_cause", "—"),
                "Category":  diag.get("question_category", "—"),
                "Outcome":   outcome_display,
                "Rolled Back": "✓ YES" if r.rolled_back else "✗ NO",
                "Time":      str(r.created_at)[:19],
            })
        render_table(pd.DataFrame(data))

        # ── Detail Inspector ──
        term_div()
        sec_header("ADAPTATION INSPECTOR")
        adapt_options = {f"Adaptation #{r.id} — Event #{r.event_id}": r.id for r in rows}
        selected_label = st.selectbox("SELECT ADAPTATION", list(adapt_options.keys()))
        selected_id = adapt_options[selected_label]
        selected = session.query(AdaptationLog).filter(AdaptationLog.id == selected_id).first()

        if selected:
            # Observation
            sec_header("OBSERVATION (WHAT WAS SEEN)")
            try:
                obs = json.loads(selected.observation or "{}")
                col1, col2, col3 = st.columns(3)
                col1.metric("SCORE BEFORE", obs.get("score_before", "—"))
                col2.metric("RET. PRECISION", f"{obs['retrieval_precision']:.2%}" if obs.get("retrieval_precision") is not None else "—")
                col3.metric("HALLUCINATION", f"{obs['hallucination_rate']:.2%}" if obs.get("hallucination_rate") is not None else "—")
                detectors = obs.get("triggered_detectors", [])
                if detectors:
                    for d in detectors:
                        st.markdown(f'<div class="flag-reason">▸ {d}</div>', unsafe_allow_html=True)
            except Exception:
                st.code(selected.observation or "No observation data")

            # Diagnosis
            sec_header("DIAGNOSIS (WHAT WAS DECIDED)")
            try:
                diag = json.loads(selected.diagnosis or "{}")
                col1, col2, col3 = st.columns(3)
                col1.metric("ROOT CAUSE", diag.get("root_cause", "—").upper())
                col2.metric("Q CATEGORY", diag.get("question_category", "—").upper())
                col3.metric("SEVERITY", diag.get("severity_score", "—"))
                reasoning = diag.get("reasoning", "")
                if reasoning:
                    st.markdown(f"""
                    <div class="answer-card">
                        <div class="answer-label">REASONING</div>
                        <div class="answer-text">{reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                st.code(selected.diagnosis or "No diagnosis data")

            # Config change
            sec_header("CONFIG CHANGE (WHAT WAS MODIFIED)")
            try:
                cfg_before = json.loads(selected.config_before or "{}")
                cfg_after = json.loads(selected.config_after or "{}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**BEFORE:**")
                    st.json(cfg_before)
                with col2:
                    st.markdown("**AFTER:**")
                    st.json(cfg_after)
            except Exception:
                pass

            # Outcome
            sec_header("OUTCOME (WHAT RESULTED)")
            try:
                met_before = json.loads(selected.metrics_before or "{}")
                met_after = json.loads(selected.metrics_after or "{}")
                col1, col2, col3 = st.columns(3)
                col1.metric("SCORE BEFORE", met_before.get("top1_score", "—"))
                col2.metric("SCORE AFTER", met_after.get("top1_score", "—"))
                col3.metric("OUTCOME", selected.outcome or "—")
            except Exception:
                pass

            if selected.rolled_back:
                st.warning("⚠ THIS ADAPTATION WAS ROLLED BACK — the change degraded metrics.")


session.close()

