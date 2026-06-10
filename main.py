"""
Self-Organising RAG — Main Application
=======================================
FastAPI application with rate-based autonomous self-healing.

The background worker uses a SLIDING WINDOW approach:
  - Every 30 seconds, checks the last 50 queries
  - If ≥15 are flagged (30% failure rate), triggers batch repair
  - Repair strategy is auto-selected per event by the DECIDE stage
"""
import asyncio
import logging
import os
from fastapi import FastAPI
from api.routes import router
from db.session import init_db, get_session
from db.models import QueryLog, LowRecallEvent
from repair.orchestrator import handle_event

# Suppress noisy HTTP client logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(title="Self-Organising RAG — Auto-RAG Pipeline")
app.include_router(router, prefix="/api/v1")

# ── Rate-based repair parameters ────────────────────────────────────
CHECK_INTERVAL_SECONDS = 30     # how often the worker checks (seconds)
RATE_WINDOW            = 50     # sliding window: look at last N queries
FAILURE_RATE_THRESHOLD = 0.30   # 30% = 15/50 flagged → trigger repair
MIN_QUERIES_BEFORE_CHECK = 10   # don't check until at least N queries exist


async def autonomous_maintenance_loop():
    """
    Rate-based self-healing: SLIDING WINDOW approach.

    Every CHECK_INTERVAL_SECONDS:
      1. Query the most recent RATE_WINDOW queries (sliding window)
      2. Count how many are flagged
      3. If failure rate >= FAILURE_RATE_THRESHOLD → batch repair all unresolved events
      4. If below threshold → do nothing (system is healthy)

    Each event's repair strategy is auto-selected by the DECIDE stage
    in repair/orchestrator.py based on:
      - Which detectors triggered (context_insufficient, hallucination_detected)
      - Query complexity (complex → bigger chunks, simple → smaller chunks)
    """
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

            session = get_session()
            try:
                # ── Sliding window: get the most recent N queries ───
                recent = session.query(QueryLog).order_by(
                    QueryLog.timestamp.desc()
                ).limit(RATE_WINDOW).all()

                if len(recent) < MIN_QUERIES_BEFORE_CHECK:
                    continue  # not enough data yet

                # ── Calculate failure rate ──────────────────────────
                flagged_count = sum(1 for q in recent if q.flagged)
                total = len(recent)
                failure_rate = flagged_count / total

                if failure_rate < FAILURE_RATE_THRESHOLD:
                    logging.info(
                        f"✅ [Rate Monitor] {flagged_count}/{total} = "
                        f"{failure_rate:.0%} — healthy (threshold: {FAILURE_RATE_THRESHOLD:.0%})"
                    )
                    continue

                # ── Rate exceeded → BATCH REPAIR ───────────────────
                logging.info(
                    f"⚠️ [Rate Monitor] {flagged_count}/{total} = "
                    f"{failure_rate:.0%} — TRIGGERING BATCH REPAIR"
                )

                unresolved = session.query(LowRecallEvent).filter(
                    LowRecallEvent.resolved == False,
                    LowRecallEvent.unfixable == False,
                ).all()

                if not unresolved:
                    logging.info("   No unresolved events to repair.")
                    continue

                logging.info(
                    f"🔧 [Batch Repair] Starting repair of {len(unresolved)} events..."
                )
                repaired, rolled_back, errors = 0, 0, 0
                for i, event in enumerate(unresolved, 1):
                    try:
                        logging.info(
                            f"   [{i}/{len(unresolved)}] Repairing event #{event.id}..."
                        )
                        result = handle_event(event.id)

                        if result.get("improved"):
                            repaired += 1
                            logging.info(
                                f"   [{i}/{len(unresolved)}] ✅ Event #{event.id} COMMITTED "
                                f"(strategy={result.get('strategy')}, "
                                f"chunk_size={result.get('chunk_size')}, "
                                f"score: {result.get('score_before', 0):.3f}→{result.get('score_after', 0):.3f})"
                            )
                        elif result.get("rolled_back"):
                            rolled_back += 1
                            logging.info(
                                f"   [{i}/{len(unresolved)}] 🔄 Event #{event.id} ROLLED BACK "
                                f"(strategy={result.get('strategy')}, "
                                f"chunk_size={result.get('chunk_size')}, "
                                f"score: {result.get('score_before', 0):.3f}→{result.get('score_after', 0):.3f})"
                            )
                        elif result.get("error"):
                            errors += 1
                            logging.warning(
                                f"   Event {event.id}: {result['error']}"
                            )
                    except Exception as e:
                        errors += 1
                        logging.error(f"   Event {event.id} repair failed: {e}")

                logging.info(
                    f"🔧 [Batch Repair] Done: "
                    f"{repaired} committed, {rolled_back} rolled back, {errors} errors"
                )

            finally:
                session.close()

        except Exception as e:
            logging.error(f"❌ [Rate Monitor] Loop error: {e}")


@app.on_event("startup")
async def startup():
    # Initialize database with latest schema
    init_db()
    logging.info("🚀 Database initialized.")

    # Check if evaluation mode
    if os.getenv("ENV") == "evaluation":
        logging.info("⏸️ Evaluation mode — background self-healing DISABLED.")
    else:
        asyncio.create_task(autonomous_maintenance_loop())
        logging.info(
            f"🤖 Rate-based self-healing worker started "
            f"(window={RATE_WINDOW}, threshold={FAILURE_RATE_THRESHOLD:.0%}, "
            f"interval={CHECK_INTERVAL_SECONDS}s)"
        )