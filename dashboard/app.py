"""
Auto-RAG — Query Diagnostic Dashboard v1
Run from test/ root: streamlit run dashboard/app.py

Month 1 Deliverable:
  ✔ Query logger visualization
  ✔ Top-K chunk retrieval with relevance scores
  ✔ Low-recall detection rules display
  ✔ Score tracking over time
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import json
import streamlit as st
import pandas as pd
from db.session import get_session
from db.models import QueryLog, LowRecallEvent, EvalSnapshot

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto-RAG Diagnostic Dashboard",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS for a polished look ───────────────────────────────────────
st.markdown("""
<style>
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: #a0a0c0 !important; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }

    /* Chunk cards */
    .chunk-card {
        background: #1a1a2e;
        border-left: 4px solid;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .chunk-card .rank { font-weight: 700; font-size: 0.8rem; margin-bottom: 6px; }
    .chunk-card .text { color: #d0d0e8; }
    .score-high   { border-color: #4ade80; }
    .score-mid    { border-color: #facc15; }
    .score-low    { border-color: #f87171; }
    .score-badge  { display: inline-block; padding: 2px 10px; border-radius: 20px;
                    font-size: 0.78rem; font-weight: 600; margin-left: 8px; }
    .badge-high   { background: #166534; color: #4ade80; }
    .badge-mid    { background: #713f12; color: #facc15; }
    .badge-low    { background: #7f1d1d; color: #f87171; }

    /* Severity badges */
    .sev-high   { color: #f87171; font-weight: 700; }
    .sev-medium { color: #facc15; font-weight: 600; }
    .sev-low    { color: #4ade80; }

    /* Section dividers */
    .section-divider { border-top: 1px solid #333355; margin: 24px 0 16px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Auto-RAG — Query Diagnostic Dashboard")

session = get_session()

# ── Sidebar navigation ───────────────────────────────────────────────────
page = st.sidebar.radio(
    "📋 Navigate",
    ["Overview", "Query Diagnostics", "Low-Recall Events", "Eval History"],
)


# ══════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    total    = session.query(QueryLog).count()
    flagged  = session.query(QueryLog).filter(QueryLog.flagged == True).count()
    healthy  = total - flagged
    events   = session.query(LowRecallEvent).count()
    resolved = session.query(LowRecallEvent).filter(LowRecallEvent.resolved == True).count()

    st.subheader("System Health")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", total)
    c2.metric("✓ Healthy", healthy)
    c3.metric("⚠ Flagged", flagged)
    c4.metric("Events", events)

    flag_rate = round(flagged / total * 100, 1) if total else 0
    res_rate  = round(resolved / events * 100, 1) if events else 0
    st.caption(f"Flag rate: **{flag_rate}%**  |  Resolution rate: **{res_rate}%**")

    # ── Score trend over time ────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("📈 Top-1 Retrieval Score Over Time")

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
            df_trend = pd.DataFrame(trend_data)
            st.line_chart(df_trend.set_index("timestamp")["top1_score"], height=250)

            # Score distribution
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Score Distribution")
                scores_all = [d["top1_score"] for d in trend_data]
                hist_series = pd.cut(pd.Series(scores_all), bins=10).value_counts().sort_index()
                hist_series.index = hist_series.index.astype(str)
                st.bar_chart(hist_series, height=200)
            with col2:
                st.subheader("Flagged vs Healthy")
                flag_counts = pd.Series({
                    "✓ Healthy": healthy,
                    "⚠ Flagged": flagged,
                })
                st.bar_chart(flag_counts, height=200)
    else:
        st.info("No queries logged yet. Send queries via `/api/v1/query` to start tracking.")

    # ── Detection rule summary ───────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("🛡 Detection Rules")
    st.markdown("""
    | Rule | Triggers When | Threshold |
    |------|---------------|-----------|
    | `low_top_score` | Best chunk similarity < threshold | **0.45** |
    | `score_drop` | Gap between rank-1 and rank-k too large | **0.3** |
    | `llm_uncertainty` | LLM says "I don't know", "unclear", etc. | keyword match |
    | `semantic_mismatch` | Retrieved chunks are semantically fragmented | **0.55** pairwise sim |
    | `evidence_mismatch` | LLM answer ≠ retrieved evidence | **0.50** cosine sim |
    | `user_frustration` | Similar query re-asked within 5 min | **0.85** cosine sim |
    """)


# ══════════════════════════════════════════════════════════════════════════
#  QUERY DIAGNOSTICS — the main Month 1 deliverable
# ══════════════════════════════════════════════════════════════════════════
elif page == "Query Diagnostics":
    st.subheader(" Query Log — Detailed Diagnostics")

    rows = session.query(QueryLog).order_by(QueryLog.timestamp.desc()).limit(100).all()

    if not rows:
        st.info("No queries logged yet. Send queries via `/api/v1/query` to populate this view.")
    else:
        # ── Summary table ────────────────────────────────────────────────
        summary_data = []
        for r in rows:
            cqs = getattr(r, "ctx_q_sim", None)
            asim = getattr(r, "answer_sem_sim", None)
            row_data = {
                "ID": r.id,
                "Query": r.query[:90],
                "Ctx↔Q Sim": round(cqs, 4) if cqs is not None else "—",
                "Answer↔GT Sim": round(asim, 4) if asim is not None else "—",
                "Status": "⚠ Flagged" if r.flagged else "✓ OK",
                "Latency (s)": f"{r.latency_ms / 1000:.2f}",
                "Time": str(r.timestamp)[:19],
            }
            summary_data.append(row_data)
        st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)

        # ── Detailed query inspector ─────────────────────────────────────
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📋 Query Inspector — Select a query to see retrieved chunks")

        query_options = {f"#{r.id} — {r.query[:70]}": r.id for r in rows}
        selected_label = st.selectbox("Select a query", list(query_options.keys()))
        selected_id = query_options[selected_label]

        selected = session.query(QueryLog).filter(QueryLog.id == selected_id).first()
        if selected:
            scores = json.loads(selected.top_k_scores or "[]")
            chunks = json.loads(selected.retrieved_chunks or "[]") if selected.retrieved_chunks else []

            # Query info header
            cqs = getattr(selected, "ctx_q_sim", None)
            asim = getattr(selected, "answer_sem_sim", None)
            has_gt = asim is not None
            if has_gt:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ctx↔Query Sim", f"{cqs:.4f}" if cqs else "N/A")
                col2.metric("Answer↔GT Sim", f"{asim:.4f}")
                col3.metric("Latency", f"{selected.latency_ms / 1000:.2f}s")
                status_text = "⚠ Flagged" if selected.flagged else "✓ Healthy"
                col4.metric("Status", status_text)
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Ctx↔Query Sim", f"{cqs:.4f}" if cqs else "N/A")
                col2.metric("Latency", f"{selected.latency_ms / 1000:.2f}s")
                status_text = "⚠ Flagged" if selected.flagged else "✓ Healthy"
                col3.metric("Status", status_text)
                st.caption("ℹ No ground truth available — Answer↔GT similarity not computed.")


            # LLM Answer
            st.markdown("**💬 LLM Answer:**")
            st.info(selected.llm_response or "_No response recorded_")



            # ── Top-K Chunks with Relevance Scores ───────────────────────
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📦 Retrieved Chunks & Relevance Scores")


            if chunks and scores:
                for i, (chunk, score) in enumerate(zip(chunks, scores)):
                    # Determine color tier
                    if score >= 0.7:
                        card_class = "score-high"
                        badge_class = "badge-high"
                        score_label = "HIGH"
                    elif score >= 0.45:
                        card_class = "score-mid"
                        badge_class = "badge-mid"
                        score_label = "MEDIUM"
                    else:
                        card_class = "score-low"
                        badge_class = "badge-low"
                        score_label = "LOW"

                    # Truncate very long chunks for display
                    display_text = chunk[:500] + ("..." if len(chunk) > 500 else "")

                    st.markdown(f"""
                    <div class="chunk-card {card_class}">
                        <div class="rank">
                            Chunk #{i+1} — Relevance: {score:.4f}
                            <span class="score-badge {badge_class}">{score_label}</span>
                        </div>
                        <div class="text">{display_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif scores:
                # Old logs without chunk content — show scores only
                st.markdown("_Chunk text not available for this query (logged before upgrade)._")
                for i, score in enumerate(scores):
                    if score >= 0.7:
                        badge = "🟢"
                    elif score >= 0.45:
                        badge = "🟡"
                    else:
                        badge = "🔴"
                    st.markdown(f"{badge} **Chunk #{i+1}** — Score: `{score:.4f}`")
            else:
                st.warning("No scores recorded for this query.")

            # ── Score bar chart for this query ───────────────────────────
            if scores:
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📊 Score Distribution for This Query")
                chart_df = pd.DataFrame({
                    "Chunk": [f"#{i+1}" for i in range(len(scores))],
                    "Relevance Score": scores,
                })
                st.bar_chart(chart_df.set_index("Chunk"), height=250)

            # ── Flagging explanation ─────────────────────────────────────
            if selected.flagged:
                event = session.query(LowRecallEvent).filter(
                    LowRecallEvent.query_log_id == selected.id
                ).first()
                if event:
                    detectors = json.loads(event.triggered_detectors or "[]")
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown("### ⚠ Why This Query Was Flagged")
                    for d in detectors:
                        explanations = {
                            "low_top_score": "🔴 **Low Top Score** — Best retrieved chunk scored below 0.45",
                            "score_drop": "🟡 **Score Drop** — Large gap between rank-1 and rank-K scores",
                            "llm_uncertainty": "🟠 **LLM Uncertainty** — Response contains hedging language",
                            "semantic_mismatch": "🟣 **Semantic Mismatch** — Retrieved chunks are about different topics",
                            "evidence_mismatch": "🔵 **Evidence Mismatch** — LLM answer doesn't match retrieved context",
                            "user_frustration": "🟤 **User Frustration** — Similar query re-asked within 5 minutes",
                        }
                        st.markdown(explanations.get(d, f"• {d}"))
                    sev_html = {
                        "HIGH": '<span class="sev-high">HIGH</span>',
                        "MEDIUM": '<span class="sev-medium">MEDIUM</span>',
                        "LOW": '<span class="sev-low">LOW</span>',
                    }
                    st.markdown(
                        f"**Severity:** {sev_html.get(event.severity, event.severity)}",
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════
#  LOW-RECALL EVENTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Low-Recall Events":
    st.subheader("⚠ Low-Recall Events")

    sev_filter = st.selectbox("Filter by severity", ["ALL", "HIGH", "MEDIUM", "LOW"])
    q = session.query(LowRecallEvent).order_by(LowRecallEvent.timestamp.desc())
    if sev_filter != "ALL":
        q = q.filter(LowRecallEvent.severity == sev_filter)
    rows = q.limit(200).all()

    if not rows:
        st.success("No low-recall events detected yet — your RAG is looking healthy! 🎉")
    else:
        data = []
        for r in rows:
            detectors = json.loads(r.triggered_detectors or "[]")
            # Fetch the original query
            log = session.query(QueryLog).filter(QueryLog.id == r.query_log_id).first()
            data.append({
                "ID": r.id,
                "Query": (log.query[:60] if log else "—"),
                "Severity": r.severity,
                "Detectors": ", ".join(detectors),
                "Resolved": "✓ Yes" if r.resolved else "✗ No",
                "Time": str(r.timestamp)[:19],
            })
        st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)




# ══════════════════════════════════════════════════════════════════════════
#  EVAL HISTORY
# ══════════════════════════════════════════════════════════════════════════
elif page == "Eval History":
    st.subheader("📊 Evaluation Snapshots")

    rows = session.query(EvalSnapshot).order_by(
               EvalSnapshot.timestamp.desc()).limit(100).all()

    if not rows:
        st.info("No evaluation snapshots yet. Run `python run_evaluation.py` to generate one.")
    else:
        data = [{
            "Namespace": r.namespace,
            "LLM": r.llm,
            "Embeddings": r.embeddings,
            "ROUGE-L": r.rouge_l,
            "Semantic Sim": r.sem_sim,
            "Ctx↔Query": r.ctx_q_sim,
            "Ctx↔Ground Truth": r.ctx_gt_sim,
            "Time": str(r.timestamp)[:19],
        } for r in rows]
        st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)

session.close()
