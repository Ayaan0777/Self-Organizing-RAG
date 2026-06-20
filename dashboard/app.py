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

from db.models import QueryLog, LowRecallEvent, EvalSnapshot, AdaptationLog, PipelineConfig, StrategyCounter, RuntimeFlag

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

/* ── Query Log Summary Row & Alignment Fixes ── */
/* Target the standard layout horizontal block container to sync grid height alignments */
div[data-testid="stHorizontalBlock"] {
    display:         flex        !important;
    align-items:     center      !important;
}

/* Push column text elements to share an identical layout cross-axis baseline */
div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
    display:         flex        !important;
    align-items:     center      !important;
}

/* Target markdown fields and standard wrappers inside the horizontal blocks to keep baselines uniform */
div[data-testid="stHorizontalBlock"] div[data-testid="column"] [data-testid="stMarkdownContainer"] p {
    margin:          0           !important;
    padding:         0           !important;
    line-height:     1.2         !important;
}

/* Clean chrome box removal for text buttons inside horizontal log blocks */
div[data-testid="stHorizontalBlock"] div[class*="st-key-ql_q_"] button {
    background:      transparent !important;
    border:          none        !important;
    border-radius:   0px         !important;
    box-shadow:      none        !important;
    outline:         none        !important;
    color:           #8899cc     !important;
    font-family:     'JetBrains Mono', monospace !important;
    font-size:       0.78rem     !important;
    font-weight:     400         !important;
    letter-spacing:  0.5px       !important;
    text-transform:  none        !important;
    text-align:      left        !important;
    text-decoration: none        !important;
    padding:         0           !important;
    margin:          0           !important;
    width:           100%        !important;
    justify-content: flex-start  !important;
    white-space:     nowrap      !important;
    overflow:        hidden      !important;
    text-overflow:   ellipsis    !important;
    cursor:          pointer     !important;
    display:         inline-flex !important;
    align-items:     center      !important;
}

/* Eliminate focus or click active states from triggering box outlines */
div[data-testid="stHorizontalBlock"] div[class*="st-key-ql_q_"] button:focus,
div[data-testid="stHorizontalBlock"] div[class*="st-key-ql_q_"] button:active {
    background:      transparent !important;
    border:          none        !important;
    outline:         none        !important;
    box-shadow:      none        !important;
}

/* Custom text transition hue effect on hovered log elements */
div[data-testid="stHorizontalBlock"] div[class*="st-key-ql_q_"] button:hover {
    color:           #00f5ff     !important;
    background:      transparent !important;
    border:          none        !important;
    box-shadow:      none        !important;
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
            elif val in ("✓ YES", "✓ RESOLVED", "✓ IMPROVED"):
                val = f'<span style="color:var(--green);">{val}</span>'
            elif val in ("✗ NO", "✗ PENDING", "✗ DEGRADED"):
                val = f'<span style="color:var(--muted);">{val}</span>'
            elif val in ("⊘ UNFIXABLE",):
                val = f'<span style="color:var(--red);font-weight:600;">{val}</span>'
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
@st.cache_resource
def _dashboard_session():
    """One SQLAlchemy session for the Streamlit process — Streamlit re-runs
    the whole script on every interaction, so without caching we'd leak one
    new session per click."""
    return get_session()


session = _dashboard_session()

page = st.sidebar.radio(
    "NAVIGATE",
    ["Overview", "Ingest Document", "Ask Query", "Add Chunks", "Query Diagnostics",
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
    resolved  = session.query(LowRecallEvent).filter(LowRecallEvent.resolved == True).count()
    unfixable = session.query(LowRecallEvent).filter(LowRecallEvent.unfixable == True).count()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("TOTAL QUERIES",  total)
    c2.metric("HEALTHY",        healthy)
    c3.metric("FLAGGED",        flagged)
    c4.metric("RESOLVED",       resolved)
    c5.metric("UNFIXABLE",      unfixable)

    flag_rate = round(flagged / total * 100, 1) if total else 0
    res_rate  = round(resolved / flagged * 100, 1) if flagged else 0
    st.caption(f"FLAG RATE: {flag_rate}%  ·  RESOLUTION RATE: {res_rate}%  ·  UNFIXABLE: {unfixable}")

    # ── Dynamic K Promotion Status ──
    dk_flag = session.query(RuntimeFlag).filter(
        RuntimeFlag.name == "dynamic_k_promoted"
    ).first()
    if dk_flag and dk_flag.value:
        st.markdown(
            '<div class="status-dot" style="margin:8px 0"><span class="dot dot-green"></span> '
            'DYNAMIC K PROMOTED — main pipeline uses category-based K selection</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-dot" style="margin:8px 0"><span class="dot dot-amber"></span> '
            'DYNAMIC K NOT YET PROMOTED — main pipeline uses fixed K=5</div>',
            unsafe_allow_html=True,
        )


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
            <td class="thresh">0.65</td>
        </tr>
        <tr>
            <td><code>score_drop</code></td>
            <td>Largest adjacent-rank score gap (K-invariant)</td>
            <td class="thresh">0.15</td>
        </tr>
        <tr>
            <td><code>llm_uncertainty</code></td>
            <td>Response contains hedging language</td>
            <td class="thresh">keyword</td>
        </tr>
        <tr>
            <td><code>semantic_mismatch</code></td>
            <td>Mean pairwise chunk sim below ratio × top1 (K-adaptive)</td>
            <td class="thresh">0.65 × top1</td>
        </tr>
        <tr>
            <td><code>evidence_mismatch</code></td>
            <td>Best chunk↔answer similarity below floor</td>
            <td class="thresh">0.60</td>
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

    uploaded_file = st.file_uploader(
        "TARGET FILE",
        type=["pdf", "docx", "txt"],
        help="Supported: PDF · DOCX · TXT",
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

    submit_query = st.button("▸ EXECUTE QUERY", type="primary", disabled=not backend_up or not query_text.strip())

    if submit_query and query_text.strip():
        with st.spinner("RETRIEVING CONTEXT · GENERATING ANSWER..."):
            try:
                payload = {"query": query_text.strip()}

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
#  ADD CHUNKS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Add Chunks":
    page_header("PIPELINE // INPUT", "ADD NEW CHUNKS")
    backend_up = backend_status()

    term_div()
    st.markdown(
        '<div style="font-family:var(--font-mono);font-size:0.8rem;color:#4a5280;margin-bottom:12px;">'
        'Paste raw text below to preview how it will be chunked. '
        'Optionally ingest chunks directly to Pinecone.'
        '</div>',
        unsafe_allow_html=True,
    )

    raw_text = st.text_area("RAW TEXT", height=200, placeholder="// paste document text here...")
    col1, col2 = st.columns([2, 1])
    with col1:
        source_name = st.text_input("SOURCE NAME", value="manual-paste")
    with col2:
        ns_chunk = st.text_input("NAMESPACE", value="", key="chunk_ns")

    col_a, col_b = st.columns(2)
    with col_a:
        chunk_btn = st.button("▸ CHUNK IT", type="primary", disabled=not raw_text.strip())
    with col_b:
        ingest_btn = st.button("▸ CHUNK + INGEST", disabled=not raw_text.strip() or not backend_up)

    if chunk_btn or ingest_btn:
        with st.spinner("Chunking..."):
            try:
                payload = {
                    "text": raw_text.strip(),
                    "source": source_name,
                    "ingest": ingest_btn,
                }
                if ns_chunk.strip():
                    payload["namespace"] = ns_chunk.strip()

                resp = requests.post(f"{API_BASE}/add-chunks", json=payload, timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    chunks_data = result.get("chunks", [])

                    sec_header(f"RESULT: {result['num_chunks']} CHUNKS ({result.get('size_range', '')})")

                    if ingest_btn and result.get("ingested"):
                        st.success(f"▸ Ingested to namespace: {result.get('namespace', 'default')}")

                    for i, c in enumerate(chunks_data):
                        score_cls = "score-high" if c['chars'] >= 500 else ("score-mid" if c['chars'] >= 200 else "score-low")
                        st.markdown(f"""
                        <div class="chunk-card {score_cls}">
                            <div class="chunk-rank">CHUNK_{i+1:02d} &nbsp;·&nbsp; {c['chars']} CHARS</div>
                            <div class="chunk-text">{c['content'][:400]}{'…' if len(c['content']) > 400 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Failed — HTTP {resp.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  QUERY DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Query Diagnostics":
    page_header("ANALYSIS // LOGS", "QUERY DIAGNOSTICS")

    # ── Pagination setup ──
    PAGE_SIZE = 10
    total_queries = session.query(QueryLog).count()
    total_pages = max(1, (total_queries + PAGE_SIZE - 1) // PAGE_SIZE)

    if "diag_page" not in st.session_state:
        st.session_state.diag_page = 0

    # Clamp page to valid range
    st.session_state.diag_page = max(0, min(st.session_state.diag_page, total_pages - 1))
    current_page = st.session_state.diag_page
    offset = current_page * PAGE_SIZE

    rows = (
        session.query(QueryLog)
        .order_by(QueryLog.timestamp.desc())
        .offset(offset)
        .limit(PAGE_SIZE)
        .all()
    )

    if not rows and total_queries == 0:
        st.info("▸ NO QUERIES LOGGED — use Ask Query page to populate this view.")
    else:
        sec_header("QUERY LOG SUMMARY")

        # ── Pagination controls ──
        nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 2, 1, 1])
        with nav1:
            if st.button("◂ PREV", disabled=(current_page == 0), key="diag_prev"):
                st.session_state.diag_page -= 1
                st.rerun()
        with nav2:
            if st.button("NEXT ▸", disabled=(current_page >= total_pages - 1), key="diag_next"):
                st.session_state.diag_page += 1
                st.rerun()
        with nav3:
            start_row = offset + 1
            end_row = min(offset + PAGE_SIZE, total_queries)
            st.markdown(
                f'<div style="font-family:var(--font-mono);font-size:0.75rem;color:var(--cyan);'
                f'letter-spacing:1px;padding:8px 0;text-align:center;">'
                f'SHOWING {start_row}–{end_row} OF {total_queries} QUERIES'
                f'</div>',
                unsafe_allow_html=True,
            )
        with nav4:
            if st.button("◂◂ FIRST", disabled=(current_page == 0), key="diag_first"):
                st.session_state.diag_page = 0
                st.rerun()
        with nav5:
            if st.button("LAST ▸▸", disabled=(current_page >= total_pages - 1), key="diag_last"):
                st.session_state.diag_page = total_pages - 1
                st.rerun()

        # ── click-to-select: track which query row was clicked ──
        if "diag_selected_query_id" not in st.session_state:
            st.session_state.diag_selected_query_id = None


        # ── Query Log Summary – column-based table; query text is the click target ──
        # Column proportions mirror the original table visually.
        _COL_W  = [0.35, 3.0, 0.85, 0.85, 0.85, 0.7, 1.3]
        _HDRS   = ["ID", "QUERY", "CTX↔Q SIM", "ANSWER↔GT", "STATUS", "LAT (s)", "TIME"]
        _TH = ('<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
               'color:#00f5ff;letter-spacing:2px;text-transform:uppercase;">%s</span>')
        _TD = ('<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;'
               'color:#8899cc;letter-spacing:0.5px;">%s</span>')

        # header row
        _hc = st.columns(_COL_W)
        for _col, _lbl in zip(_hc, _HDRS):
            _col.markdown(_TH % _lbl, unsafe_allow_html=True)

        # data rows
        for r in rows:
            cqs  = getattr(r, "ctx_q_sim",      None)
            asim = getattr(r, "answer_sem_sim", None)
            cqs_str  = f"{cqs:.4f}"  if cqs  is not None else "—"
            asim_str = f"{asim:.4f}" if asim is not None else "—"
            lat_str  = f"{r.latency_ms / 1000:.2f}"
            time_str = str(r.timestamp)[:19]
            if r.flagged:
                status_html = ('<span style="color:#ffb800;font-weight:700;'
                               'font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;">'
                               '⚠ FLAGGED</span>')
            else:
                status_html = ('<span style="color:#00ff88;'
                               'font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;">'
                               '✓ OK</span>')

            dc = st.columns(_COL_W)
            dc[0].markdown(_TD % r.id,       unsafe_allow_html=True)
            # Query cell: text layout flex container wraps the inside link trigger
            dc[1].markdown('<div class="ql-qbtn">', unsafe_allow_html=True)
            if dc[1].button(
                r.query[:120] + ("…" if len(r.query) > 120 else ""),
                key=f"ql_q_{r.id}",
            ):
                st.session_state.diag_selected_query_id = r.id
                st.rerun()
            dc[1].markdown('</div>', unsafe_allow_html=True)
            dc[2].markdown(_TD % cqs_str,   unsafe_allow_html=True)
            dc[3].markdown(_TD % asim_str,  unsafe_allow_html=True)
            dc[4].markdown(status_html,      unsafe_allow_html=True)
            dc[5].markdown(_TD % lat_str,   unsafe_allow_html=True)
            dc[6].markdown(_TD % time_str,  unsafe_allow_html=True)

        # ── Page indicator bar ──
        st.markdown(
            f'<div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--muted);'
            f'letter-spacing:2px;text-align:center;margin:8px 0 4px 0;">'
            f'PAGE {current_page + 1} / {total_pages}'
            f'</div>',
            unsafe_allow_html=True,
        )

        term_div()
        sec_header("QUERY INSPECTOR")

        query_options  = {f"#{r.id} — {r.query}": r.id for r in rows}
        _option_keys   = list(query_options.keys())
        _id_list       = list(query_options.values())
        _clicked_id    = st.session_state.get("diag_selected_query_id")
        _sel_index     = _id_list.index(_clicked_id) if _clicked_id in _id_list else 0
        selected_label = st.selectbox("SELECT QUERY", _option_keys, index=_sel_index)
        selected_id    = query_options[selected_label]
        selected        = session.query(QueryLog).filter(QueryLog.id == selected_id).first()

        if selected:
            scores = json.loads(selected.top_k_scores or "[]")
            chunks = json.loads(selected.retrieved_chunks or "[]") if selected.retrieved_chunks else []

            cqs    = getattr(selected, "ctx_q_sim", None)
            asim   = getattr(selected, "answer_sem_sim", None)
            has_gt = asim is not None

            if has_gt:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("CTX↔QUERY SIM",  f"{cqs:.4f}" if cqs is not None else "N/A")
                col2.metric("ANSWER↔GT SIM",  f"{asim:.4f}")
                col3.metric("LATENCY",          f"{selected.latency_ms / 1000:.2f}s")
                col4.metric("STATUS",           "⚠ FLAGGED" if selected.flagged else "✓ OK")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("CTX↔QUERY SIM",  f"{cqs:.4f}" if cqs is not None else "N/A")
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
                        "low_top_score":    "LOW_TOP_SCORE — best retrieved chunk scored below 0.65",
                        "score_drop":       "SCORE_DROP — largest adjacent-rank score gap exceeds 0.15 (K-invariant)",
                        "llm_uncertainty":  "LLM_UNCERTAINTY — response contains hedging language",
                        "semantic_mismatch":"SEMANTIC_MISMATCH — retrieved chunks cover different topics",
                        "evidence_mismatch":"EVIDENCE_MISMATCH — LLM answer diverges from retrieved context",
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
        res_filter = st.selectbox("FILTER BY STATUS", ["ALL", "RESOLVED", "UNRESOLVED", "UNFIXABLE"])

    # ── Build filtered query ──
    q = session.query(LowRecallEvent).order_by(LowRecallEvent.timestamp.desc())
    if sev_filter != "ALL":
        q = q.filter(LowRecallEvent.severity == sev_filter)
    if res_filter == "RESOLVED":
        q = q.filter(LowRecallEvent.resolved == True)
    elif res_filter == "UNRESOLVED":
        q = q.filter(LowRecallEvent.resolved == False, LowRecallEvent.unfixable == False)
    elif res_filter == "UNFIXABLE":
        q = q.filter(LowRecallEvent.unfixable == True)

    # ── Pagination setup ──
    PAGE_SIZE = 10
    total_events = q.count()
    total_pages = max(1, (total_events + PAGE_SIZE - 1) // PAGE_SIZE)

    if "flag_page" not in st.session_state:
        st.session_state.flag_page = 0

    # Reset to page 0 when filters change
    filter_key = f"{sev_filter}_{res_filter}"
    if st.session_state.get("flag_filter_key") != filter_key:
        st.session_state.flag_page = 0
        st.session_state.flag_filter_key = filter_key

    st.session_state.flag_page = max(0, min(st.session_state.flag_page, total_pages - 1))
    current_page = st.session_state.flag_page
    offset = current_page * PAGE_SIZE

    rows = q.offset(offset).limit(PAGE_SIZE).all()

    if not rows and total_events == 0:
        st.success("▸ NO FLAGGED EVENTS — pipeline is operating nominally.")
    else:
        # ── Pagination controls ──
        nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 2, 1, 1])
        with nav1:
            if st.button("◂ PREV", disabled=(current_page == 0), key="flag_prev"):
                st.session_state.flag_page -= 1
                st.rerun()
        with nav2:
            if st.button("NEXT ▸", disabled=(current_page >= total_pages - 1), key="flag_next"):
                st.session_state.flag_page += 1
                st.rerun()
        with nav3:
            start_row = offset + 1
            end_row = min(offset + PAGE_SIZE, total_events)
            st.markdown(
                f'<div style="font-family:var(--font-mono);font-size:0.75rem;color:var(--cyan);'
                f'letter-spacing:1px;padding:8px 0;text-align:center;">'
                f'SHOWING {start_row}–{end_row} OF {total_events} EVENTS'
                f'</div>',
                unsafe_allow_html=True,
            )
        with nav4:
            if st.button("◂◂ FIRST", disabled=(current_page == 0), key="flag_first"):
                st.session_state.flag_page = 0
                st.rerun()
        with nav5:
            if st.button("LAST ▸▸", disabled=(current_page >= total_pages - 1), key="flag_last"):
                st.session_state.flag_page = total_pages - 1
                st.rerun()

        data = []
        for r in rows:
            detectors = json.loads(r.triggered_detectors or "[]")
            log       = session.query(QueryLog).filter(QueryLog.id == r.query_log_id).first()
            status = "✓ RESOLVED" if r.resolved else ("⊘ UNFIXABLE" if r.unfixable else "✗ PENDING")
            data.append({
                "ID":        r.id,
                "Query":     (log.query if log else "—"),
                "Severity":  r.severity,
                "Detectors": ", ".join(detectors),
                "Status":    status,
                "Time":      str(r.timestamp)[:19],
            })
        render_table(pd.DataFrame(data))

        # ── Page indicator bar ──
        st.markdown(
            f'<div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--muted);'
            f'letter-spacing:2px;text-align:center;margin:8px 0 4px 0;">'
            f'PAGE {current_page + 1} / {total_pages}'
            f'</div>',
            unsafe_allow_html=True,
        )

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
                col1.metric("SCORE BEFORE", report.get("score_before") if report.get("score_before") is not None else "N/A")
                col2.metric("SCORE AFTER", report.get("score_after") if report.get("score_after") is not None else "N/A")
                col3.metric("RESOLVED STATUS", "YES" if report.get("resolved") else "NO")
                col4.metric("DYNAMIC K", report.get("dynamic_k") or "N/A")

                # ── Row 2: Strategy Explanation Card ──
                strategy_raw = report.get("strategy_used") or "unknown"
                root_cause = report.get("root_cause") or ""
                reasoning = report.get("reasoning") or ""
                q_category = report.get("question_category") or "unknown"
                chunks_before_n = report.get("chunks_before") or 0
                chunks_after_n = report.get("chunks_after") or 0

                # Map strategy to human-friendly explanation (cascade strategies)
                strategy_info = {
                    "s1_dynamic_k": {
                        "icon": "🎯", "label": "S1: DYNAMIC K SELECTION",
                        "color": "var(--cyan)",
                        "desc": "Adjusted the number of retrieved chunks (K) based on question category. "
                                "Short factual → fewer chunks (K=2–4), complex → more (K=5–8), "
                                "cross-section → most (K=6–10). No Pinecone changes.",
                    },
                    "s2_chunk_size": {
                        "icon": "🔀", "label": "S2: CHUNK SIZE VARIATION",
                        "color": "#a78bfa",
                        "desc": f"Rechunked source text with decision-engine recommended config. "
                                f"Chunks: {chunks_before_n} → {chunks_after_n}. K fixed at 5.",
                    },
                    "s3_combined": {
                        "icon": "⚡", "label": "S3: COMBINED (DYNAMIC K + RECHUNK)",
                        "color": "#f59e0b",
                        "desc": f"Applied both dynamic K selection AND rechunking together. "
                                f"Chunks: {chunks_before_n} → {chunks_after_n}.",
                    },
                    "s4_alt_llm": {
                        "icon": "🧠", "label": "S4: ALT LLM (GEMMA3:27B)",
                        "color": "#f472b6",
                        "desc": "All retrieval strategies failed. Switched to gemma3:27b (27B params) "
                                "which extracted the answer from the same chunks that mistral (7B) couldn't handle. "
                                "No Pinecone changes.",
                    },
                    "none": {
                        "icon": "❌", "label": "UNFIXABLE — ALL 4 STRATEGIES FAILED",
                        "color": "var(--red)",
                        "desc": "The cascade exhausted all 4 strategies (Dynamic K → Chunk Size → Combined → Alt LLM) "
                                "without resolving the issue. The answer likely doesn't exist in the ingested documents.",
                    },
                    # Legacy strategies (pre-cascade)
                    "semantic": {
                        "icon": "🔀", "label": "SEMANTIC RECHUNKING (LEGACY)",
                        "color": "var(--cyan)",
                        "desc": f"Re-split source document using semantic boundaries. "
                                f"Chunks: {chunks_before_n} → {chunks_after_n}.",
                    },
                }

                # Determine sub-strategy details from root cause
                root_cause_info = {
                    "chunk_too_small": {
                        "action": "EXPAND CHUNKS",
                        "detail": "Chunk size was too small — important context was split across chunks. "
                                  "Increased chunk size for more complete context per chunk.",
                    },
                    "chunk_too_large": {
                        "action": "SHRINK CHUNKS",
                        "detail": "Chunks were too large and diluted with irrelevant text. "
                                  "Reduced chunk size for more precise, focused retrieval.",
                    },
                    "general_degradation": {
                        "action": "GENERAL REPAIR",
                        "detail": "Multiple detectors fired without a clear single root cause. "
                                  "Applied general rechunking optimization.",
                    },
                    "hallucination_risk": {
                        "action": "ANTI-HALLUCINATION",
                        "detail": "LLM answer diverged from retrieved evidence. Rechunked to improve "
                                  "context precision and reduce hallucination risk.",
                    },
                    "cross_section_failure": {
                        "action": "MULTI-TOPIC REPAIR",
                        "detail": "Query spans multiple topics but chunks only covered one. "
                                  "Used topic-aware chunking to capture cross-section content.",
                    },
                }

                info = strategy_info.get(strategy_raw, {
                    "icon": "⚙️", "label": strategy_raw.upper().replace("_", " "),
                    "color": "var(--cyan)",
                    "desc": f"Applied {strategy_raw} strategy. Chunks: {chunks_before_n} → {chunks_after_n}.",
                })
                rc_info = root_cause_info.get(root_cause, None)

                term_div()
                sec_header("HOW THIS QUERY WAS RESOLVED")

                # Strategy card
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(0,245,255,0.04) 0%, rgba(0,0,0,0) 100%);
                            border: 1px solid {info['color']}40; border-left: 4px solid {info['color']};
                            border-radius: 4px; padding: 20px 24px; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                        <span style="font-size: 1.4rem;">{info['icon']}</span>
                        <span style="font-family: 'Orbitron', monospace; font-size: 0.85rem;
                                     color: {info['color']}; letter-spacing: 2px; font-weight: 700;">
                            {info['label']}
                        </span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
                                color: #8899cc; line-height: 1.6;">
                        {info['desc']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Root cause detail card (if available)
                if rc_info:
                    st.markdown(f"""
                    <div style="display: flex; gap: 16px; margin-bottom: 16px;">
                        <div style="flex: 1; background: rgba(255,150,50,0.04);
                                    border: 1px solid rgba(255,150,50,0.2);
                                    border-radius: 4px; padding: 14px 18px;">
                            <div style="font-family: 'Orbitron', monospace; font-size: 0.65rem;
                                        color: var(--amber); letter-spacing: 2px; margin-bottom: 6px;">
                                ROOT CAUSE → {rc_info['action']}
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
                                        color: #8899cc; line-height: 1.5;">
                                {rc_info['detail']}
                            </div>
                        </div>
                        <div style="flex: 1; background: rgba(0,245,255,0.04);
                                    border: 1px solid rgba(0,245,255,0.2);
                                    border-radius: 4px; padding: 14px 18px;">
                            <div style="font-family: 'Orbitron', monospace; font-size: 0.65rem;
                                        color: var(--cyan); letter-spacing: 2px; margin-bottom: 6px;">
                                QUESTION TYPE
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
                                        color: var(--cyan); font-weight: 700;">
                                {q_category.upper().replace('_', ' ')}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Reasoning (if available from diagnosis)
                if reasoning:
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.2); border: 1px solid rgba(255,150,50,0.15);
                                border-radius: 4px; padding: 12px 18px; margin-bottom: 8px;">
                        <div style="font-family: 'Orbitron', monospace; font-size: 0.6rem;
                                    color: var(--amber); letter-spacing: 2px; margin-bottom: 6px;">
                            DIAGNOSIS REASONING
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
                                    color: #8899cc; line-height: 1.5;">
                            {reasoning}
                        </div>
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

                # ── Row 4: Original Query Chunks (Fixed K=5) ──
                selected_event = session.query(LowRecallEvent).filter(
                    LowRecallEvent.id == selected_event_id
                ).first()
                original_log = None
                original_chunks = []
                if selected_event:
                    original_log = session.query(QueryLog).filter(
                        QueryLog.id == selected_event.query_log_id
                    ).first()
                    if original_log and original_log.retrieved_chunks:
                        try:
                            original_chunks = json.loads(original_log.retrieved_chunks)
                        except Exception:
                            original_chunks = []

                chunks_before = report.get("chunks_before_text") or []
                chunks_after = report.get("chunks_after_text") or []
                dyn_k = report.get("dynamic_k")

                # Deduplicate chunks
                def _dedup(chunks):
                    seen = set()
                    out = []
                    for c in chunks:
                        key = c[:200].strip()  # compare first 200 chars
                        if key not in seen:
                            seen.add(key)
                            out.append(c)
                    return out

                original_chunks = _dedup(original_chunks)
                chunks_before = _dedup(chunks_before)
                chunks_after = _dedup(chunks_after)

                if original_chunks or dyn_k or chunks_before or chunks_after:
                    term_div()

                    # ── Prominent Dynamic K Banner ──
                    repair_k = dyn_k or len(chunks_before) or "?"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(0,245,255,0.08) 0%, rgba(0,255,136,0.06) 100%);
                                border: 1px solid rgba(0,245,255,0.25); border-radius: 4px;
                                padding: 16px 24px; margin-bottom: 20px;
                                display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                        <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem;
                                    color: var(--cyan); letter-spacing: 3px; text-transform: uppercase;">
                            CHUNK RETRIEVAL COMPARISON
                        </div>
                        <div style="display: flex; align-items: center; gap: 12px;
                                    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                            <span style="background: rgba(0,245,255,0.12); border: 1px solid var(--cyan);
                                         padding: 4px 14px; border-radius: 3px; color: var(--cyan);
                                         font-weight: 700; letter-spacing: 1px;">
                                INITIAL: K = 5 (FIXED)
                            </span>
                            <span style="color: var(--muted); font-size: 1.2rem;">→</span>
                            <span style="background: rgba(0,255,136,0.12); border: 1px solid var(--green);
                                         padding: 4px 14px; border-radius: 3px; color: var(--green);
                                         font-weight: 700; letter-spacing: 1px;">
                                REPAIR: K = {repair_k} (DYNAMIC)
                            </span>
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
                                    color: var(--muted); letter-spacing: 1px; margin-left: auto;">
                            {len(chunks_before)} chunks before repair &nbsp;·&nbsp; {len(chunks_after)} after
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Original Query Chunks (Fixed K=5) ──
                    if original_chunks:
                        sec_header("ORIGINAL QUERY CHUNKS (FIXED K = 5)")
                        st.markdown("""
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                                    color: var(--muted); letter-spacing: 1px; margin-bottom: 12px;">
                            These are the 5 chunks retrieved when the query was first asked — before any repair.
                        </div>
                        """, unsafe_allow_html=True)
                        for i, chunk in enumerate(original_chunks[:5]):
                            preview = chunk[:300] + ("..." if len(chunk) > 300 else "")
                            st.markdown(f"""
                            <div style="background: rgba(0,245,255,0.03); border-left: 3px solid var(--cyan);
                                        padding: 10px 14px; margin-bottom: 6px; border-radius: 0 4px 4px 0;
                                        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
                                        color: #8899cc; line-height: 1.5;">
                                <span style="color: var(--cyan); font-weight: 600; font-size: 0.68rem;
                                             letter-spacing: 2px;">CHUNK {i+1} / 5</span><br>
                                {preview}
                            </div>
                            """, unsafe_allow_html=True)

                    # ── After Repair Chunks (Dynamic K) ──
                    term_div()
                    sec_header("REPAIR CHUNKS — AFTER REPAIR (DYNAMIC K)")

                    is_resolved = report.get("resolved", False)
                    label_color = "var(--green)" if is_resolved else "var(--red)"
                    status_label = "IMPROVED" if is_resolved else "NOT IMPROVED"
                    st.markdown(f"""
                    <div style="color: {label_color}; font-size: 0.7rem; letter-spacing: 2px;
                                margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <span style="display:inline-block; width:8px; height:8px; border-radius:50%;
                                     background: {label_color}; box-shadow: 0 0 8px {label_color};"></span>
                        AFTER REPAIR &nbsp;·&nbsp; K = {len(chunks_after)} &nbsp;·&nbsp; {status_label}
                    </div>
                    """, unsafe_allow_html=True)
                    if chunks_after:
                        for i, chunk in enumerate(chunks_after):
                            preview = chunk[:300] + ("..." if len(chunk) > 300 else "")
                            border_color = "var(--green)" if is_resolved else "var(--red)"
                            bg_color = "rgba(0,220,130,0.05)" if is_resolved else "rgba(255,60,60,0.05)"
                            st.markdown(f"""
                            <div style="background:{bg_color};border-left:3px solid {border_color};
                                        padding:10px 14px;margin-bottom:8px;border-radius:0 4px 4px 0;
                                        font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                        color:#c0c8e0;line-height:1.5;">
                                <span style="color:{border_color};font-weight:600;">CHUNK {i+1}</span><br>
                                {preview}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No chunk data recorded")

                # ── Row 5: Answers ──
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

    col1, col2, col3 = st.columns(3)
    with col1:
        new_size = st.number_input("CHUNK SIZE", min_value=100, max_value=2000, value=250, step=50)
    with col2:
        new_overlap = st.number_input("OVERLAP", min_value=0, max_value=500, value=80, step=10)
    with col3:
        new_strategy = st.selectbox("STRATEGY", ["semantic", "llm", "entropy"])

    if st.button("▸ APPLY CONFIGURATION", type="primary", disabled=not backend_up):
        try:
            params = {
                "chunk_size": new_size,
                "chunk_overlap": new_overlap,
                "chunk_strategy": new_strategy,
            }
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
    unfixable_a = session.query(AdaptationLog).filter(AdaptationLog.outcome == "UNFIXABLE").count()
    degraded_a = session.query(AdaptationLog).filter(AdaptationLog.outcome == "DEGRADED").count()
    nochange_a = session.query(AdaptationLog).filter(AdaptationLog.outcome == "NO_CHANGE").count()
    rollback_a = session.query(AdaptationLog).filter(AdaptationLog.rolled_back == True).count()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("TOTAL", total_a)
    c2.metric("IMPROVED", improved_a)
    c3.metric("UNFIXABLE", unfixable_a)
    c4.metric("DEGRADED", degraded_a)
    c5.metric("NO CHANGE", nochange_a)
    c6.metric("ROLLED BACK", rollback_a)

    if total_a:
        success_rate = round(improved_a / total_a * 100, 1)
        st.caption(f"SUCCESS RATE: {success_rate}%  ·  UNFIXABLE: {round(unfixable_a / total_a * 100, 1)}%  ·  ROLLBACK RATE: {round(rollback_a / total_a * 100, 1)}%")

    # ── Strategy Counters ──
    term_div()
    sec_header("STRATEGY SUCCESS COUNTERS")
    counters = session.query(StrategyCounter).order_by(StrategyCounter.success_count.desc()).all()
    if counters:
        counter_names = {
            "s1_dynamic_k": "🎯 S1: Dynamic K",
            "s2_chunk_size": "🔀 S2: Chunk Size",
            "s3_combined": "⚡ S3: Combined",
            "s4_alt_llm": "🧠 S4: Alt LLM",
        }
        cols = st.columns(len(counters))
        for i, c in enumerate(counters):
            label = counter_names.get(c.strategy, c.strategy.upper())
            cols[i].metric(label, c.success_count)
        # Promotion status
        dk_counter = next((c for c in counters if c.strategy == "s1_dynamic_k"), None)
        dk_count = dk_counter.success_count if dk_counter else 0
        if dk_count >= 5:
            st.markdown(
                '<div class="status-dot" style="margin:4px 0"><span class="dot dot-green"></span> '
                'S1 PROMOTED TO MAIN PIPELINE</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-dot" style="margin:4px 0"><span class="dot dot-amber"></span> '
                f'S1 PROMOTION: {dk_count}/5 successes needed</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("▸ NO STRATEGY COUNTERS — the cascade has not resolved any events yet.")

    # ── Timeline Table ──
    term_div()
    sec_header("ADAPTATION TIMELINE")
    rows = session.query(AdaptationLog).order_by(AdaptationLog.created_at.desc()).limit(50).all()

    if not rows:
        st.info("▸ NO ADAPTATIONS RECORDED — the self-healing loop has not triggered yet.")
    else:
        data = []
        for r in rows:
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
            elif outcome_display == "UNFIXABLE":
                outcome_display = "⊘ UNFIXABLE"
            elif outcome_display == "DEGRADED":
                outcome_display = "✗ DEGRADED"

            cascade_steps = obs.get("cascade_steps", [])
            steps_display = " → ".join(cascade_steps) if cascade_steps else "—"

            strategy_labels = {
                "s1_dynamic_k": "S1: Dynamic K",
                "s2_chunk_size": "S2: Chunk Size",
                "s3_combined": "S3: Combined",
                "s4_alt_llm": "S4: Alt LLM",
                "none": "None (Unfixable)",
            }

            data.append({
                "Event":     r.event_id,
                "Winner":    strategy_labels.get(r.strategy_selected, r.strategy_selected or "—"),
                "Root Cause": diag.get("root_cause", "—"),
                "Category":  diag.get("question_category", "—"),
                "Cascade":   steps_display,
                "Outcome":   outcome_display,
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
                col1, col2 = st.columns(2)
                col1.metric("SCORE BEFORE", obs.get("score_before", "—"))
                skipped_s1 = obs.get("skipped_s1", False)
                col2.metric("S1 SKIPPED", "YES (promoted)" if skipped_s1 else "NO")
                detectors = obs.get("triggered_detectors", [])
                if detectors:
                    for d in detectors:
                        st.markdown(f'<div class="flag-reason">▸ {d}</div>', unsafe_allow_html=True)
                cascade_steps = obs.get("cascade_steps", [])
                if cascade_steps:
                    sec_header("CASCADE EXECUTION")
                    for step in cascade_steps:
                        parts = step.split(":")
                        name = parts[0] if parts else step
                        result = parts[1] if len(parts) > 1 else ""
                        if result == "RESOLVED":
                            st.markdown(
                                f'<div style="background:rgba(0,255,136,0.06);border-left:3px solid var(--green);'
                                f'padding:8px 14px;margin:4px 0;border-radius:0 4px 4px 0;'
                                f'font-family:var(--font-mono);font-size:0.8rem;color:var(--green);">'
                                f'✅ {name} — RESOLVED</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div style="background:rgba(255,51,102,0.04);border-left:3px solid var(--red);'
                                f'padding:8px 14px;margin:4px 0;border-radius:0 4px 4px 0;'
                                f'font-family:var(--font-mono);font-size:0.8rem;color:var(--red);">'
                                f'❌ {name} — {result}</div>',
                                unsafe_allow_html=True,
                            )
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