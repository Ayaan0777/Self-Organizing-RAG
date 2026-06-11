"""
Batch-Driven Auto-RAG Self-Healing Daemon
============================================================
Monitors the database queue and executes repairs in batches
of 5 flagged questions to optimize computational overhead.

Uses the full decision engine pipeline:
  1. Wait until 5 flagged questions accumulate
  2. For each event: Diagnose → Select Strategy → Execute Repair
  3. Log batch summary with enhanced metrics (precision, recall, accuracy)
"""
import time
import json
import logging
from db.session import get_session
from db.models import LowRecallEvent, QueryLog
from repair.orchestrator import handle_event
from detector.decision_engine import (
    diagnose, select_strategy, check_cooldown, set_cooldown,
    get_active_config, get_recent_adaptations, STRATEGY_TO_RECHUNK,
)

BATCH_THRESHOLD = 5
POLL_INTERVAL_SECONDS = 5
MAX_ATTEMPTS = 5


def run_batch_worker():
    print("🚀 [Batch-Worker] Background Daemon initialized...")
    print(f"📊 Accumulation Mode: Waiting until queue hits {BATCH_THRESHOLD} flagged questions before execution.")
    print(f"📡 Monitoring database state every {POLL_INTERVAL_SECONDS} seconds...\n")

    while True:
        session = get_session()
        try:
            # 1. Fetch all unresolved anomalies currently in the queue
            active_anomalies = (
                session.query(LowRecallEvent)
                .filter(LowRecallEvent.resolved == False)
                .filter(LowRecallEvent.unfixable == False)
                .order_by(LowRecallEvent.timestamp.asc())
                .all()
            )

            current_queue_count = len(active_anomalies)

            # 2. Only trigger when we have accumulated enough flagged questions
            if current_queue_count >= BATCH_THRESHOLD:
                print(f"\n🔥 [THRESHOLD MET] Queue has {current_queue_count} flagged questions! "
                      f"Starting batch repair for {BATCH_THRESHOLD}...")

                # Slice the batch to process
                batch_to_process = active_anomalies[:BATCH_THRESHOLD]
                batch_results = []

                for index, event in enumerate(batch_to_process, start=1):
                    # CIRCUIT BREAKER: Check max attempts
                    if event.attempts >= MAX_ATTEMPTS:
                        logging.warning(
                            f"🛑 [Batch] Event {event.id} exhausted {MAX_ATTEMPTS} attempts. "
                            f"Marking UNFIXABLE."
                        )
                        event.unfixable = True
                        session.commit()
                        continue

                    # COOLDOWN CHECK
                    if check_cooldown(event):
                        logging.debug(f"⏳ [Batch] Event {event.id} in cooldown — skipping.")
                        continue

                    # LOAD QUERY LOG for diagnosis
                    log = session.query(QueryLog).filter(
                        QueryLog.id == event.query_log_id
                    ).first()
                    if not log:
                        logging.warning(f"[Batch] Event {event.id} has no query log — skipping.")
                        continue

                    # DIAGNOSE: Determine root cause from metrics
                    diag = diagnose(event, log)
                    print(
                        f"  ⚙️ [{index}/{BATCH_THRESHOLD}] Event {event.id} | "
                        f"Root cause: {diag['root_cause']} | "
                        f"Category: {diag['question_category']} | "
                        f"Severity: {diag['severity_score']}"
                    )

                    # SELECT STRATEGY with conflict resolution
                    recent = get_recent_adaptations(limit=5)
                    current_config = get_active_config()
                    strategy, config = select_strategy(diag, current_config, recent)
                    rechunk_strategy = STRATEGY_TO_RECHUNK.get(strategy, "semantic")

                    # Increment attempt counter
                    event.attempts += 1
                    session.commit()

                    # EXECUTE the repair with enhanced metrics
                    try:
                        result = handle_event(
                            event.id,
                            strategy=rechunk_strategy,
                            config=config,
                            diagnosis=diag,
                        )

                        batch_results.append({
                            "event_id": event.id,
                            "strategy": strategy,
                            "outcome": result.get("outcome", "ERROR"),
                            "score_before": result.get("score_before"),
                            "score_after": result.get("score_after"),
                            "precision_before": result.get("precision_before"),
                            "precision_after": result.get("precision_after"),
                            "recall_before": result.get("recall_before"),
                            "recall_after": result.get("recall_after"),
                            "accuracy_before": result.get("accuracy_before"),
                            "accuracy_after": result.get("accuracy_after"),
                        })

                        # Refresh event from DB to avoid stale session data
                        session.expire(event)

                        if result.get("improved"):
                            print(f"    ✅ Event {event.id} HEALED using {strategy}.")
                            event.resolved = True
                            session.commit()
                        else:
                            status = "ROLLED_BACK" if result.get("rolled_back") else "FAILED"
                            print(f"    ⚠️ Event {event.id} {status}. Will re-diagnose next cycle.")
                            event.resolved = False  # Explicitly keep unresolved
                            set_cooldown(event, session)

                    except Exception as loop_err:
                        print(f"    ❌ Error processing Event {event.id}: {str(loop_err)}")
                        batch_results.append({
                            "event_id": event.id,
                            "strategy": strategy,
                            "outcome": "ERROR",
                            "error": str(loop_err),
                        })

                # ── BATCH SUMMARY ──
                improved_count = sum(1 for r in batch_results if r.get("outcome") == "IMPROVED")
                rolled_back_count = sum(1 for r in batch_results if r.get("outcome") == "DEGRADED")
                error_count = sum(1 for r in batch_results if r.get("outcome") == "ERROR")

                print(f"\n🏁 [Batch Complete] Processed {len(batch_results)} events:")
                print(f"   ✅ Improved: {improved_count}")
                print(f"   🔄 Rolled Back: {rolled_back_count}")
                print(f"   ❌ Errors: {error_count}")

                # Log detailed metrics for each repair
                for r in batch_results:
                    if r.get("precision_before") is not None:
                        print(
                            f"   Event {r['event_id']}: "
                            f"prec={r.get('precision_before', 'N/A')}→{r.get('precision_after', 'N/A')} "
                            f"recall={r.get('recall_before', 'N/A')}→{r.get('recall_after', 'N/A')} "
                            f"acc={r.get('accuracy_before', 'N/A')}→{r.get('accuracy_after', 'N/A')} "
                            f"[{r['outcome']}]"
                        )

                print(f"   Returning to accumulation mode.\n")

            else:
                # Still accumulating — show heartbeat
                if current_queue_count > 0:
                    print(
                        f"⏳ Accumulating: {current_queue_count}/{BATCH_THRESHOLD} "
                        f"flagged questions in queue. Standing by...",
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