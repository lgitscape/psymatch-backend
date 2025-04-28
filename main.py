# 📦 main.py

from fastapi import FastAPI
from prometheus_client import Counter, start_http_server, REGISTRY
from pydantic_settings import BaseSettings
import structlog
import uvicorn

from api.handlers import router as api_router
from api.handlers import initialize_explainer
from supabase_client import supabase
from utils.supabase_utils import insert_with_retry
from engine.models import init_lambda_model
from utils.fetch_therapists import fetch_therapists
from pydantic import Field

log = structlog.get_logger()

# ─────────────────────────────
# Settings
class Settings(BaseSettings):
    app_name: str = "PsyMatch Recommender"
    version: str = "5.5.5"
    host: str = "0.0.0.0"
    port: int = Field(8000, env="PORT")
    prometheus_port: int = Field(0, env="PROMETHEUS_PORT")

settings = Settings()

# ─────────────────────────────
# Prometheus monitoring

if "psymatch_requests" not in REGISTRY._names:
    REQUEST_COUNTER = Counter("psymatch_requests", "Total /recommend requests made")

if "psymatch_fallbacks" not in REGISTRY._names:
    FALLBACK_COUNTER = Counter("psymatch_fallbacks", "Total number of fallbacks to rule-based scoring")

if "psymatch_matches_returned" not in REGISTRY._names:
    MATCHES_RETURNED_COUNTER = Counter("psymatch_matches_returned", "Number of matches returned per request")

if "psymatch_matches_filtered_out" not in REGISTRY._names:
    FILTERED_OUT_COUNTER = Counter("psymatch_matches_filtered_out", "Number of matches filtered out under minimum score")

# ─────────────────────────────
# Dummy therapist list (later replace with DB fetch)
THERAPISTS = []

# ─────────────────────────────
# API Setup
app = FastAPI(title=settings.app_name, version=settings.version)
app.include_router(api_router)

# ─────────────────────────────
# Startup event
@app.on_event("startup")
async def startup_event():
    global THERAPISTS
    THERAPISTS = await fetch_therapists()
    start_http_server(settings.prometheus_port)
    try:
        from engine.model_registry_helper import get_latest_production_model
        model_path = get_latest_production_model() or "./models/latest_model.txt"
        init_lambda_model(model_path)
        initialize_explainer()
    except Exception as e:
        log.warning("Could not initialize LambdaRank model", error=str(e))

# ─────────────────────────────
# Main entrypoint
if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
    )