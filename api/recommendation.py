from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path
from engine.model import load_models, predict_ensemble
from engine.features import build_feature_vector
from supabase_client import supabase
from schemas import ClientProfile, RecommendResponse, ErrorResponse
from utils import find_therapist_by_id, insert_with_retry
from config import THERAPISTS
import structlog

log = structlog.get_logger()
router = APIRouter()

# Load ensemble models once at startup
try:
    models = load_models(Path("models/ensemble_models_latest.pkl"))  # Dynamisch kiezen kan later
    log.info("Loaded ensemble models successfully")
except Exception as e:
    models = None
    log.warning("Failed to load models, fallback to matcher only", error=str(e))


@router.post("/recommend", response_model=RecommendResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def recommend(client: ClientProfile, top_n: int = 10):
    if not client.topics or not client.languages or not client.timeslots:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(status="error", message="Topics, languages and timeslots are required.").model_dump()
        )

    matches, algorithm_used = await run_matcher(client, THERAPISTS, top_n=top_n)

    if not matches:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(status="error", message="No suitable therapists found with at least 80% match score.").model_dump()
        )

    if models:
        try:
            features_list = []
            for match in matches:
                therapist = find_therapist_by_id(match["therapist_id"])
                if therapist is None:
                    continue
                fv = build_feature_vector(client, therapist)
                features_list.append(fv)

            X_new = pd.DataFrame(features_list)

            # Zorg dat categoricals goed worden geïnterpreteerd
            for col in ["item_category_1", "item_category_2", "item_category_3"]:
                if col in X_new.columns:
                    X_new[col] = X_new[col].astype("category")

            preds = predict_ensemble(models, X_new)

            # Voeg predicted success scores toe
            for match, pred in zip(matches, preds):
                match["predicted_success_score"] = pred

            # Sorteren op predicted success
            matches = sorted(matches, key=lambda x: x["predicted_success_score"], reverse=True)

            algorithm_used += "+ML"
            log.info("Successfully reranked matches with ML model")

        except Exception as e:
            log.error("Failed to rerank matches with ML model", error=str(e))

    try:
        best_match = matches[0]
        row = {
            "client_id": str(client.client_id),
            "algorithm": algorithm_used,
            "top_match_id": best_match["therapist_id"],
            "top_match_normalized_score": best_match["score_normalized"],
            "predicted_success_score": best_match.get("predicted_success_score"),
            "recommended": matches,
        }
        insert_with_retry(supabase.table("match_logs"), row)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(status="error", message="Failed to log match to Supabase.", info=str(e)).model_dump()
        )

    return RecommendResponse(status="success", data=matches)
