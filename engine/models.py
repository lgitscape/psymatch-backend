# 📦 engine/models.py
# ─────────────────────────────
# ML model loader for LightGBM Regression Ensemble

from pathlib import Path
import joblib
import pandas as pd
import structlog

log = structlog.get_logger()

class LightGBMModel:
    def __init__(self, models):
        self.models = models

    def predict(self, feature_df: pd.DataFrame):
        preds = [model.predict(feature_df) for model in self.models]
        return sum(preds) / len(preds)

# Global singleton
lightgbm_model: LightGBMModel | None = None

def load_lightgbm_model(version: str = None) -> None:
    """
    Loads the LightGBM ensemble model.
    If version is None, loads the latest available model.
    """
    global lightgbm_model
    model_dir = Path("models")

    if version:
        model_file = model_dir / f"ensemble_models_{version}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"No model found for version {version}")
    else:
        model_files = list(model_dir.glob("ensemble_models_*.pkl"))
        if not model_files:
            raise FileNotFoundError("No ensemble model files found.")
        model_file = max(model_files, key=lambda p: p.stat().st_ctime)

    models = joblib.load(model_file)
    lightgbm_model = LightGBMModel(models)
    log.info("LightGBM model loaded", path=str(model_file))
