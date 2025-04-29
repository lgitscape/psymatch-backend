# 📦 engine/models/lambda_model.py
# ─────────────────────────────────
# Ensemble model loader for PsyMatch v5.5.0+

import lightgbm as lgb
from pathlib import Path

class LambdaModel:
    def __init__(self, model_paths):
        self.models = [lgb.Booster(model_file=str(path)) for path in model_paths]

    def predict(self, X):
        """Predict by averaging ensemble predictions."""
        preds = [model.predict(X) for model in self.models]
        return sum(preds) / len(preds)

# Load ensemble models
model_dir = Path("models/")
model_paths = sorted(model_dir.glob("model_fold*.txt"))
lambda_model = LambdaModel(model_paths) if model_paths else None
