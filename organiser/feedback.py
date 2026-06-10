"""
Feedback Loop Analyser — Phase 4: Self-Organising RAG
======================================================
Analyses the effectiveness of detection rules and repair strategies
over time. Provides actionable insights for system tuning.

This module is READ-ONLY — it analyses data and suggests changes
but never auto-modifies thresholds or system behaviour.
"""
import json
import logging
from collections import Counter


def analyse_system_health() -> dict:
    """
    Comprehensive analysis of the detection + repair pipeline's effectiveness.

    Returns a report with:
      - detector_stats:   how often each detector fires, false-positive estimates
      - strategy_stats:   per-strategy success rates and score improvements
      - threshold_suggestions: recommended threshold adjustments
      - overall_health:   summary metrics

    This is exposed via GET /api/v1/feedback/analysis
    """
    try:
        from db.session import get_session
        from db.models import QueryLog, LowRecallEvent, RepairReport

        session = get_session()
        try:
            # ── 1. Query-level stats ──
            total_queries = session.query(QueryLog).count()
            flagged_queries = session.query(QueryLog).filter(QueryLog.flagged == True).count()
            healthy_queries = total_queries - flagged_queries

            # ── 2. Detector fire rates ──
            events = session.query(LowRecallEvent).all()
            detector_counts = Counter()
            severity_counts = Counter()

            for event in events:
                try:
                    detectors = json.loads(event.triggered_detectors or "[]")
                    for d in detectors:
                        detector_counts[d] += 1
                except (json.JSONDecodeError, TypeError):
                    pass
                severity_counts[event.severity] += 1

            detector_stats = []
            for detector_name, count in detector_counts.most_common():
                fire_rate = round(count / total_queries * 100, 2) if total_queries > 0 else 0.0
                detector_stats.append({
                    "detector": detector_name,
                    "times_fired": count,
                    "fire_rate_pct": fire_rate,
                })

            # ── 3. Event resolution stats ──
            total_events = len(events)
            resolved_events = sum(1 for e in events if e.resolved)
            unfixable_events = sum(1 for e in events if getattr(e, "unfixable", False))
            pending_events = total_events - resolved_events - unfixable_events

            # ── 4. Repair strategy stats ──
            from organiser.adaptive_strategy import get_strategy_stats
            strategy_stats = get_strategy_stats()

            # ── 5. Threshold suggestions ──
            suggestions = _compute_threshold_suggestions(
                total_queries, flagged_queries, detector_stats, resolved_events, total_events
            )

            # ── 6. Overall health score (0-100) ──
            health_score = _compute_health_score(
                total_queries, flagged_queries, resolved_events, total_events
            )

            return {
                "overall": {
                    "health_score": health_score,
                    "total_queries": total_queries,
                    "healthy_queries": healthy_queries,
                    "flagged_queries": flagged_queries,
                    "flag_rate_pct": round(flagged_queries / total_queries * 100, 2) if total_queries > 0 else 0.0,
                },
                "events": {
                    "total": total_events,
                    "resolved": resolved_events,
                    "unfixable": unfixable_events,
                    "pending": pending_events,
                    "resolution_rate_pct": round(resolved_events / total_events * 100, 2) if total_events > 0 else 0.0,
                },
                "severity_distribution": dict(severity_counts),
                "detector_stats": detector_stats,
                "strategy_stats": strategy_stats,
                "threshold_suggestions": suggestions,
            }

        finally:
            session.close()

    except Exception as e:
        logging.error(f"[feedback] analysis failed: {e}")
        return {"error": str(e)}


def _compute_threshold_suggestions(
    total_queries: int,
    flagged_queries: int,
    detector_stats: list[dict],
    resolved_events: int,
    total_events: int,
) -> list[dict]:
    """
    Generates threshold adjustment suggestions based on detection patterns.
    These are informational — they don't auto-apply.
    """
    suggestions = []

    # High flag rate might mean thresholds are too aggressive
    flag_rate = flagged_queries / total_queries * 100 if total_queries > 0 else 0
    if flag_rate > 50:
        suggestions.append({
            "type": "warning",
            "message": f"Flag rate is high ({flag_rate:.1f}%). "
                       "Consider raising SCORE_LOW threshold from 0.45 to 0.40 "
                       "to reduce false positives.",
            "detector": "low_top_score",
            "current_threshold": 0.45,
            "suggested_threshold": 0.40,
        })

    # Low flag rate might mean thresholds are too lenient
    if total_queries > 20 and flag_rate < 5:
        suggestions.append({
            "type": "info",
            "message": f"Flag rate is very low ({flag_rate:.1f}%). "
                       "System may be missing poor-quality responses. "
                       "Consider lowering SCORE_LOW threshold from 0.45 to 0.50.",
            "detector": "low_top_score",
            "current_threshold": 0.45,
            "suggested_threshold": 0.50,
        })

    # If a specific detector fires excessively
    for stat in detector_stats:
        if stat["fire_rate_pct"] > 30:
            suggestions.append({
                "type": "warning",
                "message": f"Detector '{stat['detector']}' fires on {stat['fire_rate_pct']}% of queries. "
                           "Consider tuning its threshold or reviewing ingested content quality.",
                "detector": stat["detector"],
            })

    # Low resolution rate suggests repair strategies aren't effective
    resolution_rate = resolved_events / total_events * 100 if total_events > 0 else 0
    if total_events > 5 and resolution_rate < 30:
        suggestions.append({
            "type": "warning",
            "message": f"Repair resolution rate is low ({resolution_rate:.1f}%). "
                       "Consider re-ingesting source documents with better chunking, "
                       "or adding more diverse content.",
        })

    if not suggestions:
        suggestions.append({
            "type": "info",
            "message": "System is operating within normal parameters. No adjustments recommended.",
        })

    return suggestions


def _compute_health_score(
    total_queries: int,
    flagged_queries: int,
    resolved_events: int,
    total_events: int,
) -> int:
    """
    Computes an overall health score (0-100) for the RAG pipeline.

    Factors:
      - Low flag rate → higher score
      - High resolution rate → higher score
      - Having enough data → avoids extreme scores
    """
    if total_queries == 0:
        return 50  # neutral — no data

    # Flag rate component (0-50 points): lower flag rate = better
    flag_rate = flagged_queries / total_queries
    flag_score = max(0, 50 - int(flag_rate * 100))

    # Resolution rate component (0-50 points): higher resolution = better
    if total_events > 0:
        resolution_rate = resolved_events / total_events
        resolution_score = int(resolution_rate * 50)
    else:
        resolution_score = 50  # no events = no problems

    return min(100, flag_score + resolution_score)
