# 📦 main.py

from fastapi import FastAPI
from prometheus_client import Counter, start_http_server
from pydantic import BaseSettings
import structlog
import uvicorn

from api.handlers import router as api_router
from api.handlers import initialize_explainer
from supabase_client import supabase
from utils.supabase_utils import insert_with_retry
from engine.models import init_lambda_model
from utils.fetch_therapists import fetch_therapists

log = structlog.get_logger()

# ─────────────────────────────
# Settings
class Settings(BaseSettings):
    app_name: str = "PsyMatch Recommender"
    version: str = "5.5.5"
    prometheus_port: int = 8001
    host: str = "0.0.0.0"
    port: int = 8000

settings = Settings()

# ─────────────────────────────
# Prometheus monitoring
REQUEST_COUNTER = Counter("psymatch_requests_total", "Total /recommend requests made")
FALLBACK_COUNTER = Counter("psymatch_fallbacks_total", "Total number of fallbacks to rule-based scoring")
MATCHES_RETURNED_COUNTER = Counter("psymatch_matches_returned", "Number of matches returned per request")
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
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
