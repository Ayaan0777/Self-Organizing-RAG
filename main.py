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
    Runs continuously in the background using metric-driven strategy selection.
    
    Replaces the old fixed waterfall (semantic → entropy → llm) with:
      1. Cooldown check — skip events still in cooldown
      2. Diagnosis — analyze which metrics failed and why
      3. Strategy selection — pick the best fix with conflict resolution
      4. Execution — repair with dynamic chunk config + rollback safety
    """
    CHECK_INTERVAL_SECONDS = 10  # Keep at 10 for testing
    MAX_ATTEMPTS = 5  # More attempts since strategies are now intelligent
    
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            session = get_session()
            try:
                # Only grab events that are unresolved AND not marked unfixable
                unresolved_events = session.query(LowRecallEvent).filter(
                    LowRecallEvent.resolved == False,
                    LowRecallEvent.unfixable == False
                ).all()
                
                if unresolved_events:
                    logging.info(f"⚠️ [Auto-Worker] Found {len(unresolved_events)} pending events.")
                    
                    for event in unresolved_events:
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
                            f"🔍 [Auto-Worker] Event {event.id} | "
                            f"Root cause: {diagnosis['root_cause']} | "
                            f"Category: {diagnosis['question_category']} | "
                            f"Severity: {diagnosis['severity_score']}"
                        )

                        # SELECT STRATEGY with conflict resolution
                        recent = get_recent_adaptations(limit=5)
                        current_config = get_active_config()
                        strategy, config = select_strategy(diagnosis, current_config, recent)
                        
                        # Map the strategy to the rechunk function name
                        rechunk_strategy = STRATEGY_TO_RECHUNK.get(strategy, "semantic")
                        
                        logging.info(
                            f"⚙️ [Auto-Worker] Event {event.id} | "
                            f"Attempt {event.attempts + 1}/{MAX_ATTEMPTS} | "
                            f"Strategy: {strategy} → rechunk: {rechunk_strategy} | "
                            f"Config: size={config.get('chunk_size')} overlap={config.get('chunk_overlap')}"
                        )
                        
                        # Increment the attempt counter immediately
                        event.attempts += 1
                        session.commit()

                        # EXECUTE the repair with dynamic config
                        result = handle_event(
                            event.id,
                            strategy=rechunk_strategy,
                            config=config,
                            diagnosis=diagnosis,
                        )
                        
                        if result.get("improved"):
                            logging.info(f"✨ [Auto-Worker] Success! Event {event.id} repaired using {strategy}.")
                            event.resolved = True
                            session.commit()
                        else:
                            logging.info(
                                f"📉 [Auto-Worker] Strategy '{strategy}' "
                                f"{'rolled back' if result.get('rolled_back') else 'failed'} "
                                f"on Event {event.id}. Will re-diagnose next cycle."
                            )
                            # Set cooldown to prevent immediate re-attempt
                            set_cooldown(event, session)
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