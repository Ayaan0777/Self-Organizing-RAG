"""
Batch-Driven Auto-RAG Self-Healing Daemon
============================================================
Monitors the database queue and triggers the repair cascade
when the failure threshold is met.

Trigger condition (BOTH must be true):
  - Pending count ≥ PENDING_MIN (5)
  - Pending / total queries ≥ PENDING_RATIO (30%)

Each event gets exactly one cascade pass (S1→S2→S3→S4).
No retries, no cooldowns — single-pass with unfixable terminal state.
"""
import time
import logging
from db.session import get_session
from db.models import LowRecallEvent, QueryLog

PENDING_MIN = 5
PENDING_RATIO = 0.30
POLL_INTERVAL_SECONDS = 5


def run_batch_worker():
    print("🚀 [Batch-Worker] Background Daemon initialized...")
    print(f"📊 Trigger: ≥{PENDING_MIN} pending events AND ≥{int(PENDING_RATIO*100)}% of total queries flagged.")
    print(f"📡 Monitoring database state every {POLL_INTERVAL_SECONDS} seconds...\n")

    while True:
        session = get_session()
        try:
            # 1. Count total queries (denominator for ratio)
            total_queries = session.query(QueryLog).count()

            # 2. Fetch all pending events (unresolved AND not unfixable)
            pending_events = (
                session.query(LowRecallEvent)
                .filter(LowRecallEvent.resolved == False)
                .filter(LowRecallEvent.unfixable == False)
                .order_by(LowRecallEvent.timestamp.asc())
                .all()
            )
            pending_count = len(pending_events)

            # 3. Check trigger conditions
            count_ok = pending_count >= PENDING_MIN
            ratio_ok = total_queries > 0 and (pending_count / total_queries) >= PENDING_RATIO

            if count_ok and ratio_ok:
                ratio_pct = round(pending_count / total_queries * 100, 1)
                print(f"\n🔥 [THRESHOLD MET] {pending_count} pending events "
                      f"({ratio_pct}% of {total_queries} queries). "
                      f"Starting cascade for ALL {pending_count} events...")

                # Import cascade here to avoid circular imports at module load
                from repair.cascade import run_repair_cascade

                batch_results = []

                for index, event in enumerate(pending_events, start=1):
                    print(f"\n  ⚙️ [{index}/{pending_count}] Processing Event {event.id}...")

                    try:
                        result = run_repair_cascade(event.id)
                        batch_results.append({
                            "event_id": event.id,
                            "winning_strategy": result.get("winning_strategy"),
                            "outcome": result.get("outcome", "ERROR"),
                            "cascade_steps": result.get("cascade_steps", []),
                            "score_before": result.get("score_before"),
                            "score_after": result.get("score_after"),
                        })

                        status = result.get("outcome", "ERROR")
                        winner = result.get("winning_strategy", "none")
                        if status == "IMPROVED":
                            print(f"    ✅ Event {event.id} HEALED by {winner}")
                        elif status == "UNFIXABLE":
                            print(f"    ❌ Event {event.id} UNFIXABLE (all 4 strategies failed)")
                        else:
                            print(f"    ⚠️ Event {event.id}: {status}")

                    except Exception as loop_err:
                        print(f"    ❌ Error processing Event {event.id}: {str(loop_err)}")
                        batch_results.append({
                            "event_id": event.id,
                            "outcome": "ERROR",
                            "error": str(loop_err),
                        })

                # ── BATCH SUMMARY ──
                improved_count = sum(1 for r in batch_results if r.get("outcome") == "IMPROVED")
                unfixable_count = sum(1 for r in batch_results if r.get("outcome") == "UNFIXABLE")
                error_count = sum(1 for r in batch_results if r.get("outcome") == "ERROR")

                print(f"\n🏁 [Batch Complete] Processed {len(batch_results)} events:")
                print(f"   ✅ Resolved: {improved_count}")
                print(f"   ❌ Unfixable: {unfixable_count}")
                print(f"   ⚠️ Errors: {error_count}")

                # Per-event detail
                for r in batch_results:
                    steps = " → ".join(r.get("cascade_steps", []))
                    print(f"   Event {r['event_id']}: {r.get('outcome')} "
                          f"| Winner: {r.get('winning_strategy', 'none')} "
                          f"| Steps: [{steps}]")

                print(f"   Returning to monitoring mode.\n")

            else:
                # Still accumulating — show heartbeat
                if pending_count > 0:
                    if total_queries > 0:
                        ratio_pct = round(pending_count / total_queries * 100, 1)
                    else:
                        ratio_pct = 0.0
                    reasons = []
                    if not count_ok:
                        reasons.append(f"count={pending_count}/{PENDING_MIN}")
                    if not ratio_ok:
                        reasons.append(f"ratio={ratio_pct}%/{int(PENDING_RATIO*100)}%")
                    print(
                        f"⏳ Waiting: {pending_count} pending events, "
                        f"but threshold not met ({', '.join(reasons)})",
                        end="\r"
                    )

        except Exception as db_err:
            print(f"\n❌ Daemon database lookup error: {str(db_err)}")
        finally:
            session.close()

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run_batch_worker()
    except KeyboardInterrupt:
        print("\n🛑 Background batch daemon shut down cleanly.")