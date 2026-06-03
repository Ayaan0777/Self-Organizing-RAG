from fastapi import FastAPI
from api.routes import router
from db.session import init_db

app = FastAPI(title="Self-Organising RAG — Auto-RAG Pipeline")
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def startup():
    init_db()