import asyncio
import logging
from fastapi import FastAPI
from api.routes import router
from db.session import init_db, get_session
from db.models import LowRecallEvent, QueryLog
from repair.orchestrator import handle_event
from detector.decision_engine import (
    diagnose, select_strategy, check_cooldown, set_cooldown,
    get_active_config, get_recent_adaptations, STRATEGY_TO_RECHUNK,
)
import os
# Set up logging format
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(title="Self-Organising RAG — Auto-RAG Pipeline")
app.include_router(router, prefix="/api/v1")

async def autonomous_maintenance_loop():
    """
    Batch-driven self-healing loop — only triggers after 5 flagged questions
    accumulate. Processes them as a batch with full diagnosis + strategy
    selection + enhanced metric validation (precision, recall, accuracy).
    """
    CHECK_INTERVAL_SECONDS = 10
    BATCH_THRESHOLD = 5
    MAX_ATTEMPTS = 5
    
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            session = get_session()
            try:
                # Only grab events that are unresolved AND not marked unfixable
                unresolved_events = session.query(LowRecallEvent).filter(
                    LowRecallEvent.resolved == False,
                    LowRecallEvent.unfixable == False
                ).order_by(LowRecallEvent.timestamp.asc()).all()

                current_count = len(unresolved_events)

                # BATCH GATE: Only process when threshold is met
                if current_count < BATCH_THRESHOLD:
                    if current_count > 0:
                        logging.debug(
                            f"⏳ [Auto-Worker] Accumulating: {current_count}/{BATCH_THRESHOLD} "
                            f"flagged questions. Standing by..."
                        )
                    continue

                logging.info(
                    f"🔥 [Auto-Worker] Threshold met! {current_count} flagged questions. "
                    f"Processing batch of {BATCH_THRESHOLD}..."
                )

                batch = unresolved_events[:BATCH_THRESHOLD]
                improved_count = 0
                rolled_back_count = 0

                for idx, event in enumerate(batch, start=1):
                    # CIRCUIT BREAKER: Check if we've exhausted max attempts
                    if event.attempts >= MAX_ATTEMPTS:
                        logging.warning(f"🛑 [Auto-Worker] Event {event.id} exhausted {MAX_ATTEMPTS} attempts. Marking UNFIXABLE.")
                        event.unfixable = True
                        session.commit()
                        continue

                    # COOLDOWN CHECK: Skip events still in cooldown period
                    if check_cooldown(event):
                        logging.debug(f"⏳ [Auto-Worker] Event {event.id} in cooldown — skipping.")
                        continue

                    # LOAD QUERY LOG for diagnosis
                    log = session.query(QueryLog).filter(
                        QueryLog.id == event.query_log_id
                    ).first()
                    if not log:
                        logging.warning(f"[Auto-Worker] Event {event.id} has no query log — skipping.")
                        continue

                    # DIAGNOSE: Determine root cause from metrics
                    diagnosis = diagnose(event, log)
                    logging.info(
                        f"⚙️ [Auto-Worker] [{idx}/{BATCH_THRESHOLD}] Event {event.id} | "
                        f"Root cause: {diagnosis['root_cause']} | "
                        f"Category: {diagnosis['question_category']} | "
                        f"Severity: {diagnosis['severity_score']}"
                    )

                    # SELECT STRATEGY with conflict resolution
                    recent = get_recent_adaptations(limit=5)
                    current_config = get_active_config()
                    strategy, config = select_strategy(diagnosis, current_config, recent)
                    rechunk_strategy = STRATEGY_TO_RECHUNK.get(strategy, "semantic")

                    # Increment the attempt counter immediately
                    event.attempts += 1
                    session.commit()

                    # EXECUTE the repair with enhanced metrics
                    result = handle_event(
                        event.id,
                        strategy=rechunk_strategy,
                        config=config,
                        diagnosis=diagnosis,
                    )
                    
                    if result.get("improved"):
                        logging.info(f"✅ [Auto-Worker] Event {event.id} HEALED using {strategy}.")
                        event.resolved = True
                        session.commit()
                        improved_count += 1
                    else:
                        logging.info(
                            f"⚠️ [Auto-Worker] Strategy '{strategy}' "
                            f"{'rolled back' if result.get('rolled_back') else 'failed'} "
                            f"on Event {event.id}. Will re-diagnose next cycle."
                        )
                        set_cooldown(event, session)
                        if result.get("rolled_back"):
                            rolled_back_count += 1

                logging.info(
                    f"🏁 [Auto-Worker] Batch complete: "
                    f"✅ {improved_count} improved, 🔄 {rolled_back_count} rolled back. "
                    f"Returning to accumulation mode."
                )
            finally:
                session.close()

        except Exception as e:
            logging.error(f"❌ [Auto-Worker] Loop error: {e}")

@app.on_event("startup")
async def startup():
    # 1. Initialize the fresh database with the correct new columns
    init_db()
    logging.info("🚀 Database layout initialized successfully.")
    
    # 2. Check if we are running in evaluation mode
    if os.getenv("ENV") == "evaluation":
        logging.info("⏸️ [Auto-Worker] Evaluation mode detected. Background self-healing is DISABLED.")
    else:
        # 3. Normal mode: Detach the self-healing worker
        asyncio.create_task(autonomous_maintenance_loop())
        logging.info("🤖 Autonomous self-healing worker detached and running.")