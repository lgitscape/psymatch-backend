# 📦 engine/models.py
# ─────────────────────────────
# ML model loader for LambdaRank (LightGBM)

from lightgbm import LGBMRanker
from pathlib import Path
import structlog
import pandas as pd

log = structlog.get_logger()

class LambdaRankModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = LGBMRanker()
            self.model.booster_.load_model(self.model_path)
            log.info("LambdaRank model loaded", path=self.model_path)
        except Exception as e:
            log.error("Failed to load LambdaRank model", error=str(e))
            self.model = None

    def predict(self, feature_df: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model.predict(feature_df)

# Global singleton
lambda_model: LambdaRankModel | None = None

def init_lambda_model(model_path: str) -> None:
    global lambda_model
    lambda_model = LambdaRankModel(model_path)
