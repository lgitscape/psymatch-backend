from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, BaseSettings
from prometheus_client import Counter, start_http_server
import uuid
import structlog
from engine.matcher import Matcher
from supabase_client import supabase
import shap
from shap import Explainer
import pandas as pd
from typing import List, Optional
from fastapi.responses import JSONResponse
from engine.models import init_lambda_model
from utils.supabase_utils import insert_with_retry

log = structlog.get_logger()

# ─────────────────────────────
# Settings
class Settings(BaseSettings):
    app_name: str = "PsyMatch Recommender"
    version: str = "5.5.3"
    prometheus_port: int = 8001
    host: str = "0.0.0.0"
    port: int = 8000

settings = Settings()

# ─────────────────────────────
# Dummy therapist list (later fetchen uit database)
THERAPISTS = []

# ─────────────────────────────
# Prometheus monitoring
REQUEST_COUNTER = Counter("psymatch_requests_total", "Total /recommend requests made")
FALLBACK_COUNTER = Counter("psymatch_fallbacks_total", "Total number of fallbacks to rule-based scoring")
MATCHES_RETURNED_COUNTER = Counter("psymatch_matches_returned", "Number of matches returned per request")
FILTERED_OUT_COUNTER = Counter("psymatch_matches_filtered_out", "Number of matches filtered out under minimum score")

# ─────────────────────────────
# Input and Output models

class ClientProfile(BaseModel):
    client_id: uuid.UUID
    setting: str
    max_km: int
    topics: List[str]
    topic_weights: dict[str, int]
    style_pref: str
    style_weight: int
    gender_pref: Optional[str] = None
    therapy_goals: List[str] = []
    client_traits: List[str] = []
    languages: List[str]
    timeslots: List[str]
    budget: Optional[float] = None
    severity: int = 3
    lat: Optional[float] = None
    lon: Optional[float] = None

class MatchResult(BaseModel):
    therapist_id: str
    score_raw: float
    score_normalized: float

class RecommendResponse(BaseModel):
    status: str
    data: List[MatchResult]

class ExplainResponse(BaseModel):
    status: str
    data: dict

class ChooseMatchResponse(BaseModel):
    status: str
    message: str
    match_id: str
    therapist_id: str

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    version: str

class ErrorResponse(BaseModel):
    status: str
    message: str
    info: Optional[str | dict] = None

# ─────────────────────────────
# Dependency Injection

def get_supabase():
    return supabase

# ─────────────────────────────
# API Setup

app = FastAPI(title=settings.app_name, version=settings.version)

@app.on_event("startup")
async def startup_event():
    global THERAPISTS
    THERAPISTS = await fetch_therapists()
    start_http_server(settings.prometheus_port)
    try:
        from engine.model_registry_helper import get_latest_production_model
        model_path = get_latest_production_model() or "./models/latest_model.txt"
        init_lambda_model(model_path)
    except Exception as e:
        log.warning("Could not initialize LambdaRank model", error=str(e))

@app.get("/", response_model=HealthCheckResponse)
async def healthcheck():
    return JSONResponse(content=HealthCheckResponse(
        status="ok",
        message="PsyMatch matching engine live",
        version=settings.version
    ).model_dump())

@app.post("/recommend", response_model=RecommendResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def recommend(client: ClientProfile, top_n: int = 10, supabase=Depends(get_supabase)):
    REQUEST_COUNTER.inc()

    if not client.topics or not client.languages or not client.timeslots:
        raise HTTPException(status_code=422, detail="Topics, languages and timeslots are required.")

    matcher = Matcher(client, THERAPISTS)
    matches, algorithm_used = await matcher.run(top_n=top_n)

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(status="error", message="No suitable therapists found with at least 80% match score.").model_dump()
        )

    try:
        best_match = matches[0]
        row = {
            "client_id": str(client.client_id),
            "algorithm": algorithm_used,
            "top_match_id": best_match["therapist_id"],
            "top_match_normalized_score": best_match["score_normalized"],
            "recommended": matches,
        }
        insert_with_retry(supabase.table("match_logs"), row)

    except Exception as e:
        raise HTTPException(status_code=500, detail=ErrorResponse(status="error", message="Failed to log match to Supabase.", info=str(e)).model_dump())

    return RecommendResponse(status="success", data=matches)

@app.post("/match/{match_id}/choose/{therapist_id}", response_model=ChooseMatchResponse, responses={500: {"model": ErrorResponse}})
async def choose_match(match_id: str, therapist_id: str, supabase=Depends(get_supabase)):
    try:
        response = supabase.table("match_logs").select("id").eq("id", match_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail=f"Match ID {match_id} not found.")

        update_response = supabase.table("match_logs").update({"chosen_match_id": therapist_id}).eq("id", match_id).execute()

        if update_response.status_code >= 400:
            raise Exception(update_response.data)

        return ChooseMatchResponse(
            status="success",
            message="Therapist chosen successfully.",
            match_id=match_id,
            therapist_id=therapist_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=ErrorResponse(status="error", message="Failed to choose match.", info=str(e)).dict())

@app.post("/explain", response_model=ExplainResponse, responses={500: {"model": ErrorResponse}})
async def explain_ep(client: ClientProfile):
    from engine.features import build_feature_vector
    from shap import TreeExplainer
    import pandas as pd

    try:
        if not THERAPISTS:
            raise HTTPException(status_code=400, detail="No therapists available to explain.")

        matcher = Matcher(client, THERAPISTS)
        matches, _ = await matcher.run(top_n=1)

        if not matches:
            raise HTTPException(status_code=404, detail="No suitable therapist found to explain.")

        best_match_id = matches[0]['therapist_id']
        best_match_therapist = next((th for th in THERAPISTS if th.id == best_match_id), None)

        if not best_match_therapist:
            raise HTTPException(status_code=404, detail="Matched therapist not found in database.")

        feat = build_feature_vector(client, best_match_therapist)
        feat_df = pd.DataFrame([feat])

        if feat_df.empty:
            raise HTTPException(status_code=400, detail="Feature vector is empty.")

        if not lambda_model or not lambda_model.model:
            raise HTTPException(status_code=500, detail="Model not loaded.")

        explainer = TreeExplainer(lambda_model.model.booster_)
        shap_values = explainer.shap_values(feat_df)

        explanation = {
            "base_value": float(explainer.expected_value),
            "feature_values": feat_df.iloc[0].to_dict(),
            "shap_values": dict(zip(feat_df.columns, [round(val, 4) for val in shap_values[0]])),
            "positive_impact_features": [col for col, val in zip(feat_df.columns, shap_values[0]) if val > 0],
            "negative_impact_features": [col for col, val in zip(feat_df.columns, shap_values[0]) if val < 0],
        }

        return ExplainResponse(status="success", data=explanation)

    except Exception as e:
        raise HTTPException(status_code=500, detail=ErrorResponse(status="error", message="Failed to explain match.", info=str(e)).model_dump())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
