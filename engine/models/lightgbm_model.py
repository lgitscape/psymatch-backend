# 📦 engine/models/lightgbm_model.py

from pathlib import Path
import lightgbm as lgb

class LightGBMModel:
    def __init__(self, model_paths):
        self.models = [lgb.Booster(model_file=str(path)) for path in model_paths]

    def predict(self, X):
        """Predict by averaging ensemble predictions."""
        preds = [model.predict(X) for model in self.models]
        return sum(preds) / len(preds)

# Global model instance
lightgbm_model = None

def load_lightgbm_model(version=None, model_path=None):
    """Load the LightGBM model(s).

    Args:
        version (str, optional): If provided, load models from versioned subfolder.
        model_path (str or Path, optional): Specific model file path.
    """
    global lightgbm_model

    if model_path:
        # If a direct model path is given, use it
        model_paths = [Path(model_path)]
    elif version:
        # If a version is given, load from that versioned subfolder
        model_dir = Path(f"models/{version}/")
        model_paths = sorted(model_dir.glob("model_fold*.txt"))
    else:
        # Default: load from models root
        model_dir = Path("models/")
        model_paths = sorted(model_dir.glob("model_fold*.txt"))

    if model_paths:
        lightgbm_model = LightGBMModel(model_paths)
    else:
        lightgbm_model = None
