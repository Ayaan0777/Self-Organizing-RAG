"""
Adaptive Strategy Selector — Phase 4: Self-Organising RAG
==========================================================
Replaces the hard-coded repair strategy waterfall with a data-driven
approach. Analyses past RepairReport outcomes to determine which
strategies work best, and returns them in optimal order.

Safety: Returns the default order ["semantic", "entropy", "llm"]
if no historical data exists.
"""
import logging

ALL_STRATEGIES = ["semantic", "entropy", "llm"]
DEFAULT_ORDER = ["semantic", "entropy", "llm"]


def get_optimal_strategy_order() -> list[str]:
    """
    Analyses historical RepairReports to determine the best strategy order.

    Logic:
      1. Query all RepairReports from the database
      2. For each strategy, compute: success_rate = resolved / total
      3. Sort strategies by success rate (highest first)
      4. Append any strategies with no history at the end

    Returns:
        list[str] — Strategy names ordered by historical success rate.
                    e.g. ["entropy", "semantic", "llm"]
    """
    try:
        from db.session import get_session
        from db.models import RepairReport

        session = get_session()
        try:
            reports = session.query(RepairReport).all()

            if not reports:
                logging.info("[adaptive] no repair history — using default order")
                return DEFAULT_ORDER.copy()

            # Aggregate stats per strategy
            stats: dict[str, dict] = {}
            for r in reports:
                s = r.strategy_used
                if s not in stats:
                    stats[s] = {"total": 0, "resolved": 0, "total_improvement": 0.0}
                stats[s]["total"] += 1
                if r.resolved:
                    stats[s]["resolved"] += 1
                # Track average score improvement regardless of resolved flag
                if r.score_before is not None and r.score_after is not None:
                    stats[s]["total_improvement"] += (r.score_after - r.score_before)

            # Compute success rate and average improvement
            rated = []
            for strategy, data in stats.items():
                if strategy not in ALL_STRATEGIES:
                    continue  # ignore unknown strategies
                success_rate = data["resolved"] / data["total"] if data["total"] > 0 else 0.0
                avg_improvement = data["total_improvement"] / data["total"] if data["total"] > 0 else 0.0
                rated.append({
                    "strategy": strategy,
                    "success_rate": success_rate,
                    "avg_improvement": avg_improvement,
                    "total": data["total"],
                    "resolved": data["resolved"],
                })

            # Sort by success rate (primary), then by avg improvement (secondary)
            rated.sort(key=lambda x: (x["success_rate"], x["avg_improvement"]), reverse=True)

            # Build ordered list, append missing strategies at the end
            ordered = [item["strategy"] for item in rated]
            for s in ALL_STRATEGIES:
                if s not in ordered:
                    ordered.append(s)

            logging.info(
                f"[adaptive] strategy order: {ordered} "
                f"(based on {len(reports)} historical reports)"
            )
            return ordered

        finally:
            session.close()

    except Exception as e:
        logging.warning(f"[adaptive] failed to compute optimal order ({e}), using default")
        return DEFAULT_ORDER.copy()


def get_strategy_stats() -> list[dict]:
    """
    Returns detailed per-strategy statistics for the feedback endpoint.

    Returns:
        List of dicts with keys: strategy, total, resolved, success_rate, avg_improvement
    """
    try:
        from db.session import get_session
        from db.models import RepairReport

        session = get_session()
        try:
            reports = session.query(RepairReport).all()

            stats: dict[str, dict] = {}
            for r in reports:
                s = r.strategy_used
                if s not in stats:
                    stats[s] = {"total": 0, "resolved": 0, "total_improvement": 0.0}
                stats[s]["total"] += 1
                if r.resolved:
                    stats[s]["resolved"] += 1
                if r.score_before is not None and r.score_after is not None:
                    stats[s]["total_improvement"] += (r.score_after - r.score_before)

            result = []
            for strategy, data in stats.items():
                result.append({
                    "strategy": strategy,
                    "total": data["total"],
                    "resolved": data["resolved"],
                    "success_rate": round(data["resolved"] / data["total"], 4) if data["total"] > 0 else 0.0,
                    "avg_improvement": round(data["total_improvement"] / data["total"], 4) if data["total"] > 0 else 0.0,
                })

            result.sort(key=lambda x: x["success_rate"], reverse=True)
            return result

        finally:
            session.close()

    except Exception as e:
        logging.warning(f"[adaptive] failed to get stats: {e}")
        return []
