import asyncio
import logging
from fastapi import FastAPI
from api.routes import router
from db.session import init_db, get_session
from db.models import LowRecallEvent
from repair.orchestrator import handle_event
import os
# Set up logging format
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(title="Self-Organising RAG — Auto-RAG Pipeline")
app.include_router(router, prefix="/api/v1")

async def autonomous_maintenance_loop():
    """Runs continuously in the background using a Multi-Strategy Waterfall."""
    CHECK_INTERVAL_SECONDS = 10  # Keep at 10 for testing
    
    # The 3 strategies the worker will cycle through
    STRATEGY_WATERFALL = ["semantic", "entropy", "llm"]
    
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            session = get_session()
            try:
                # 🛡️ THE NEW FILTER: Only grab events that are unresolved AND not marked unfixable
                unresolved_events = session.query(LowRecallEvent).filter(
                    LowRecallEvent.resolved == False,
                    LowRecallEvent.unfixable == False
                ).all()
                
                if unresolved_events:
                    logging.info(f"⚠️ [Auto-Worker] Found {len(unresolved_events)} pending events.")
                    
                    for event in unresolved_events:
                        # CIRCUIT BREAKER: Check if we've exhausted all strategies
                        if event.attempts >= len(STRATEGY_WATERFALL):
                            logging.warning(f"🛑 [Auto-Worker] Event {event.id} exhausted all {len(STRATEGY_WATERFALL)} attempts. Marking UNFIXABLE.")
                            event.unfixable = True
                            session.commit()
                            continue

                        # WATERFALL ROUTING: Pick the strategy based on the attempt count
                        current_strategy = STRATEGY_WATERFALL[event.attempts]
                        
                        logging.info(f"⚙️ [Auto-Worker] Event {event.id} | Attempt {event.attempts + 1}/3 | Strategy: {current_strategy}")
                        
                        # Increment the attempt counter immediately so we don't get stuck if it crashes
                        event.attempts += 1
                        session.commit()

                        # Execute the repair surgery
                        result = handle_event(event.id, strategy=current_strategy)
                        
                        if result.get("improved"):
                            logging.info(f"✨ [Auto-Worker] Success! Event {event.id} repaired using {current_strategy}.")
                            event.resolved = True
                            session.commit()
                        else:
                            logging.info(f"📉 [Auto-Worker] Strategy '{current_strategy}' failed on Event {event.id}. Will escalate next cycle.")
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