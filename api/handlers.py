# 📦 /api/handlers.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from schemas.schemas import (
    ClientProfile,
    RecommendResponse,
    ErrorResponse,
    ExplainResponse,
    ChooseMatchResponse,
    HealthCheckResponse,
)
from services.matcher_service import (
    run_matcher,
    run_explanation,
    REQUEST_COUNTER,
    FALLBACK_COUNTER,
    MATCHES_RETURNED_COUNTER,
    FILTERED_OUT_COUNTER,
)
from supabase_client import supabase
from utils.supabase_utils import insert_with_retry
from engine.models import lambda_model
import structlog
import pandas as pd
from engine.features import build_feature_vector
import shap
import asyncio
from scripts.setup_training_data import main as setup_training_data
from scripts.retrain_model import main as retrain_model_main

log = structlog.get_logger()

router = APIRouter()

# Placeholder therapist list — updated at startup
THERAPISTS = []

# Create SHAP explainer globally
explainer = None

# ─────────────────────────────
def initialize_explainer():
    global explainer
    if lambda_model and lambda_model.model:
        explainer = shap.TreeExplainer(lambda_model.model.booster_)

# ─────────────────────────────
# Health Check
@router.get("/", response_model=HealthCheckResponse)
async def healthcheck():
    return JSONResponse(content=HealthCheckResponse(
        status="ok",
        message="PsyMatch matching engine live",
        version="5.5.3"
    ).model_dump())

# ─────────────────────────────
# Recommend Endpoint
@router.post("/recommend", response_model=RecommendResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def recommend(client: ClientProfile, top_n: int = 10):
    REQUEST_COUNTER.inc()

    if not client.topics or not client.languages or not client.timeslots:
        raise HTTPException(status_code=422, detail="Topics, languages and timeslots are required.")

    matches, algorithm_used = await run_matcher(client, THERAPISTS, top_n=top_n)

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

# ─────────────────────────────
# Explain Endpoint
@router.post("/explain", response_model=ExplainResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def explain(client: ClientProfile):
    match = await run_explanation(client, THERAPISTS)

    if not match:
        raise HTTPException(status_code=404, detail="No suitable therapist found for explanation.")

    therapist_id = match["therapist_id"]
    best_match_therapist = next((th for th in THERAPISTS if th.id == therapist_id), None)

    if not best_match_therapist:
        raise HTTPException(status_code=404, detail="Therapist not found.")

    feat = build_feature_vector(client, best_match_therapist)
    feat_df = pd.DataFrame([feat])

    if not lambda_model or not lambda_model.model:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    global explainer

    if not explainer:
        raise HTTPException(status_code=500, detail="Explainer not initialized")

    shap_values = await asyncio.to_thread(explainer.shap_values, feat_df)


    explanation = {
        "base_value": float(explainer.expected_value),
        "feature_values": feat_df.iloc[0].to_dict(),
        "shap_values": dict(zip(feat_df.columns, [round(val, 4) for val in shap_values[0]])),
        "positive_impact_features": [col for col, val in zip(feat_df.columns, shap_values[0]) if val > 0],
        "negative_impact_features": [col for col, val in zip(feat_df.columns, shap_values[0]) if val < 0],
    }

    sorted_shap = sorted(zip(feat_df.columns, shap_values[0]), key=lambda x: -abs(x[1]))
    top_features = [col for col, val in sorted_shap[:5]]

    explanation.update({
        "top_features_by_impact": top_features
    })

    return ExplainResponse(status="success", data=explanation)

# ─────────────────────────────
# Choose Match Endpoint
@router.post("/match/{match_id}/choose/{therapist_id}", response_model=ChooseMatchResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def choose_match(match_id: str, therapist_id: str):
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
        raise HTTPException(status_code=500, detail=ErrorResponse(status="error", message="Failed to choose match.", info=str(e)).model_dump())

# ─────────────────────────────
# Model Feature Importance
@router.get("/model/feature-importance", response_model=dict)
async def feature_importance():
    if not lambda_model or not lambda_model.model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    booster = lambda_model.model.booster_
    importance = booster.feature_importance(importance_type='gain')
    feature_names = booster.feature_name()
    return {
        "feature_importance": dict(zip(feature_names, importance.tolist()))
    }

# ─────────────────────────────
# Mock Model Retrain
@router.post("/model/retrain", response_model=dict)
async def retrain_model():
    """Trigger retraining pipeline (mocked but future-proof)."""
    # In the future: trigger a real MLflow / LightGBM pipeline here
    log.info("Retraining pipeline triggered (future real pipeline).")
    await asyncio.sleep(1)  # simulate some work
    return {"status": "success", "message": "Retraining pipeline started (mock)."}

# ─────────────────────────────────────────────────────────────
# Admin Endpoints
# ─────────────────────────────────────────────────────────────

@router.post("/admin/setup-training-data")
async def admin_setup_training_data():
    """Genereer fake therapists en training matches in Supabase."""
    setup_training_data()
    return {"status": "setup complete"}

@router.post("/admin/train-model")
async def admin_train_model():
    """Train LambdaRank model op Supabase-data en sla op."""
    retrain_model_main()
    return {"status": "model trained"}

from engine.models import init_lambda_model

@router.post("/admin/reload-model")
async def admin_reload_model():
    """Reload het nieuw getrainde LambdaRank model in het geheugen."""
    init_lambda_model("models/latest_model.txt")
    return {"status": "model reloaded"}
