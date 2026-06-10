"""
Decision Engine — Stage 3 (DECIDE)
====================================
Metric-driven strategy selection that replaces the fixed waterfall.

Components:
  1. diagnose()          — Analyzes which metrics failed and determines root cause
  2. select_strategy()   — Picks the best remediation strategy considering
                           priority, conflicts, and cooldown
  3. check_cooldown()    — Prevents oscillation by enforcing wait periods

Decision Rules:
  Low precision on short factual Qs  → reduce_chunk_size  (chunk 512→256)
  Insufficient context on complex Qs → increase_chunk_size (chunk 250→512)
  Cross-section retrieval failures   → large_coherent_chunks (chunk 1024+)
  High hallucination rate            → tighten_chunks (chunk→200, low overlap)
  Stale content drift                → re_ingest (trigger auto_indexer)
"""
import json
import logging
from datetime import datetime, timedelta

from db.session import get_session
from db.models import QueryLog, LowRecallEvent, AdaptationLog, PipelineConfig


# ── Strategy configuration tables ──
# Each strategy maps to a recommended chunk configuration change.
STRATEGY_CONFIGS = {
    "reduce_chunk_size": {
        "chunk_size": 256,
        "chunk_overlap": 50,
        "chunk_strategy": "semantic",
        "description": "Smaller chunks to reduce dilution on short factual queries",
    },
    "increase_chunk_size": {
        "chunk_size": 512,
        "chunk_overlap": 120,
        "chunk_strategy": "semantic",
        "description": "Larger chunks with more overlap for complex multi-part queries",
    },
    "large_coherent_chunks": {
        "chunk_size": 1024,
        "chunk_overlap": 200,
        "chunk_strategy": "semantic",
        "description": "Very large chunks to preserve cross-paragraph coherence",
    },
    "tighten_chunks": {
        "chunk_size": 200,
        "chunk_overlap": 40,
        "chunk_strategy": "semantic",
        "description": "Tight, precise chunks to minimize noise and hallucination",
    },
    "re_ingest": {
        "chunk_size": None,   # keep current
        "chunk_overlap": None,
        "chunk_strategy": None,
        "description": "Re-embed stale content without changing chunk parameters",
    },
}

# Maps rechunk strategy names to the repair/chunker.py function names
STRATEGY_TO_RECHUNK = {
    "reduce_chunk_size": "semantic",
    "increase_chunk_size": "semantic",
    "large_coherent_chunks": "semantic",
    "tighten_chunks": "semantic",
    "re_ingest": "semantic",
}

# Default cooldown: seconds to wait before re-attempting repair on the same source
DEFAULT_COOLDOWN_SECONDS = 120


# ══════════════════════════════════════════════════════════════
#  1. DIAGNOSE — Determine root cause from metrics & triggers
# ══════════════════════════════════════════════════════════════

def diagnose(event, query_log) -> dict:
    """
    Analyzes a LowRecallEvent and its associated QueryLog to determine
    the root cause and recommend a remediation strategy.

    Examines:
      - Which detectors fired (triggered_detectors)
      - The query's question category (short_factual / complex / cross_section)
      - Stage 2 metrics (retrieval_precision, context_sufficiency, hallucination_rate)
      - Retrieval scores from top-K

    Returns:
        {
            "root_cause": str,
            "question_category": str,
            "severity_score": float (0–1),
            "recommended_strategy": str,
            "recommended_config": dict,
            "reasoning": str,
        }
    """
    triggers = json.loads(event.triggered_detectors or "[]")
    q_category = query_log.question_category or _infer_category(query_log.query)
    scores = json.loads(query_log.top_k_scores or "[]")

    # Gather available metrics
    ret_precision = query_log.retrieval_precision
    ctx_suff = query_log.context_sufficiency
    hall_rate = query_log.hallucination_rate

    # ── Scoring: calculate severity based on number and type of triggers ──
    severity_score = min(len(triggers) / 6.0, 1.0)  # 6 max detectors

    # ── Root cause diagnosis using metric correlation ──
    root_cause = "unknown"
    reasoning = ""

    # Priority 1: High hallucination → tighten chunks
    if (hall_rate is not None and hall_rate > 0.3) or "evidence_mismatch" in triggers:
        root_cause = "high_hallucination"
        reasoning = (f"Hallucination rate {hall_rate or 'N/A'} exceeds threshold. "
                     f"Evidence mismatch detected. Chunks likely contain too much "
                     f"irrelevant text confusing the LLM.")
        severity_score = max(severity_score, 0.8)

    # Priority 2: Low precision on short factual queries → reduce chunk size
    elif q_category == "short_factual" and (
        "low_top_score" in triggers or
        (ret_precision is not None and ret_precision < 0.4)
    ):
        root_cause = "chunk_too_large"
        reasoning = (f"Short factual query with low precision ({ret_precision or 'N/A'}). "
                     f"Chunks are likely too large, diluting the relevant content.")
        severity_score = max(severity_score, 0.7)

    # Priority 3: Cross-section failure → large coherent chunks
    elif q_category == "cross_section" and "semantic_mismatch" in triggers:
        root_cause = "cross_section_failure"
        reasoning = (f"Cross-section query with semantically fragmented retrieval. "
                     f"Document coherence is lost across chunks.")
        severity_score = max(severity_score, 0.75)

    # Priority 4: Insufficient context on complex queries → increase chunk size
    elif q_category == "complex" and (
        (ctx_suff is not None and ctx_suff == False) or
        "llm_uncertainty" in triggers
    ):
        root_cause = "chunk_too_small"
        reasoning = (f"Complex query with insufficient context (sufficiency={ctx_suff}). "
                     f"Chunks are too small to capture the full answer.")
        severity_score = max(severity_score, 0.6)

    # Priority 5: Score drop → possible stale content
    elif "score_drop" in triggers and scores and scores[0] < 0.4:
        root_cause = "stale_content"
        reasoning = (f"Large score drop with very low top score ({scores[0]:.3f}). "
                     f"Embeddings may be stale or content has drifted.")
        severity_score = max(severity_score, 0.5)

    # Default fallback: use low_top_score as generic signal
    elif "low_top_score" in triggers:
        root_cause = "chunk_too_large"
        reasoning = "Low retrieval score — defaulting to smaller chunks for better precision."
        severity_score = max(severity_score, 0.5)

    # If nothing specific, try semantic rechunk as a general repair
    else:
        root_cause = "general_degradation"
        reasoning = f"Multiple detectors triggered ({triggers}) without a clear single cause."

    # Map root cause → strategy
    strategy_map = {
        "high_hallucination": "tighten_chunks",
        "chunk_too_large": "reduce_chunk_size",
        "cross_section_failure": "large_coherent_chunks",
        "chunk_too_small": "increase_chunk_size",
        "stale_content": "re_ingest",
        "general_degradation": "reduce_chunk_size",
    }
    recommended = strategy_map.get(root_cause, "reduce_chunk_size")
    config = STRATEGY_CONFIGS[recommended]

    return {
        "root_cause": root_cause,
        "question_category": q_category,
        "severity_score": round(severity_score, 3),
        "recommended_strategy": recommended,
        "recommended_config": {
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "chunk_strategy": config["chunk_strategy"],
        },
        "reasoning": reasoning,
    }


def _infer_category(query: str) -> str:
    """Fallback question classification when category isn't pre-computed."""
    from controllers.metrics import classify_question
    return classify_question(query)


# ══════════════════════════════════════════════════════════════
#  2. SELECT STRATEGY — With conflict resolution & prioritization
# ══════════════════════════════════════════════════════════════

def select_strategy(
    diagnosis: dict,
    current_config: dict = None,
    recent_adaptations: list = None,
) -> tuple:
    """
    Selects the best strategy considering:
      1. The diagnosis recommendation
      2. Conflict resolution — don't do the opposite of what we just did
      3. Avoid repeating the same strategy that already failed

    Args:
        diagnosis: Output of diagnose()
        current_config: Current PipelineConfig as dict
        recent_adaptations: Recent AdaptationLog entries for this source

    Returns:
        (strategy_name: str, config: dict)
    """
    recommended = diagnosis["recommended_strategy"]
    config = diagnosis["recommended_config"]
    recent_adaptations = recent_adaptations or []

    # ── Conflict resolution ──
    # Check if the recommended strategy contradicts a recent one
    conflicting_pairs = {
        ("reduce_chunk_size", "increase_chunk_size"),
        ("increase_chunk_size", "reduce_chunk_size"),
        ("tighten_chunks", "large_coherent_chunks"),
        ("large_coherent_chunks", "tighten_chunks"),
    }

    for adaptation in recent_adaptations[-3:]:  # check last 3 adaptations
        prev_strategy = adaptation.strategy_selected
        if (recommended, prev_strategy) in conflicting_pairs:
            # Previous adaptation did the opposite — check if it helped
            if adaptation.outcome == "IMPROVED":
                # The opposite actually helped — don't reverse it
                logging.warning(
                    f"[decision] Conflict: '{recommended}' contradicts recent "
                    f"'{prev_strategy}' which IMPROVED metrics. Keeping current config."
                )
                return prev_strategy, _get_config_dict(prev_strategy, current_config)

    # ── Avoid repeating a recently-failed strategy ──
    for adaptation in recent_adaptations[-3:]:
        if (adaptation.strategy_selected == recommended and
                adaptation.outcome in ("DEGRADED", "NO_CHANGE")):
            logging.info(
                f"[decision] Strategy '{recommended}' recently failed. "
                f"Trying fallback."
            )
            # Fallback: if reducing failed, try tightening; if increasing failed, try large
            fallbacks = {
                "reduce_chunk_size": "tighten_chunks",
                "increase_chunk_size": "large_coherent_chunks",
                "tighten_chunks": "reduce_chunk_size",
                "large_coherent_chunks": "increase_chunk_size",
                "re_ingest": "reduce_chunk_size",
            }
            fallback = fallbacks.get(recommended, "reduce_chunk_size")
            return fallback, STRATEGY_CONFIGS[fallback]

    return recommended, config


def _get_config_dict(strategy: str, current_config: dict = None) -> dict:
    """Gets the config dict for a strategy, preserving current values for re_ingest."""
    config = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["reduce_chunk_size"])
    if strategy == "re_ingest" and current_config:
        return {
            "chunk_size": current_config.get("chunk_size", 250),
            "chunk_overlap": current_config.get("chunk_overlap", 80),
            "chunk_strategy": current_config.get("chunk_strategy", "semantic"),
        }
    return {
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "chunk_strategy": config["chunk_strategy"],
    }


# ══════════════════════════════════════════════════════════════
#  3. COOLDOWN — Prevent oscillation
# ══════════════════════════════════════════════════════════════

def check_cooldown(event, session=None) -> bool:
    """
    Returns True if the event is still in a cooldown period,
    meaning we should SKIP this event and not attempt repair yet.

    Cooldown prevents the system from flip-flopping between strategies
    before enough data accumulates to judge the previous change.
    """
    if event.cooldown_until and event.cooldown_until > datetime.utcnow():
        logging.debug(
            f"[decision] Event {event.id} in cooldown until "
            f"{event.cooldown_until.isoformat()}"
        )
        return True
    return False


def set_cooldown(event, session, cooldown_seconds: int = None):
    """Sets a cooldown period on an event after a repair attempt."""
    seconds = cooldown_seconds or DEFAULT_COOLDOWN_SECONDS
    event.cooldown_until = datetime.utcnow() + timedelta(seconds=seconds)
    event.last_repair_at = datetime.utcnow()
    session.commit()


# ══════════════════════════════════════════════════════════════
#  4. HELPERS — Pipeline config management
# ══════════════════════════════════════════════════════════════

def get_active_config(namespace: str = None) -> dict:
    """
    Gets the current active PipelineConfig for a namespace.
    Returns default values if no config exists yet.
    """
    session = get_session()
    try:
        query = session.query(PipelineConfig).filter(
            PipelineConfig.active == True
        )
        if namespace:
            query = query.filter(PipelineConfig.namespace == namespace)

        config = query.order_by(PipelineConfig.created_at.desc()).first()

        if config:
            return {
                "id": config.id,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "chunk_strategy": config.chunk_strategy,
                "namespace": config.namespace,
            }

        # Default config if none exists
        return {
            "id": None,
            "chunk_size": 250,
            "chunk_overlap": 80,
            "chunk_strategy": "semantic",
            "namespace": namespace,
        }
    finally:
        session.close()


def save_new_config(namespace: str, chunk_size: int, chunk_overlap: int,
                    chunk_strategy: str = "semantic") -> int:
    """
    Saves a new PipelineConfig and deactivates the old one.
    Returns the new config ID.
    """
    session = get_session()
    try:
        # Deactivate all previous configs for this namespace
        session.query(PipelineConfig).filter(
            PipelineConfig.namespace == namespace,
            PipelineConfig.active == True
        ).update({"active": False})

        new_config = PipelineConfig(
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
            active=True,
        )
        session.add(new_config)
        session.commit()
        session.refresh(new_config)
        logging.info(
            f"[decision] Saved new config: size={chunk_size} overlap={chunk_overlap} "
            f"strategy={chunk_strategy}"
        )
        return new_config.id
    except Exception as e:
        session.rollback()
        logging.error(f"[decision] Failed to save config: {e}")
        return -1
    finally:
        session.close()


def get_recent_adaptations(event_id: int = None, limit: int = 5) -> list:
    """Gets recent AdaptationLog entries for conflict resolution."""
    session = get_session()
    try:
        query = session.query(AdaptationLog).order_by(
            AdaptationLog.created_at.desc()
        ).limit(limit)
        return query.all()
    finally:
        session.close()
