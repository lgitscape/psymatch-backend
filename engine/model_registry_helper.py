# model_registry_helper.py – LambdaRank model path resolver
# ========================================================
"""
Return a file-system path to the most recent *production* LightGBM LambdaRank
model so that `recommender.py` can hot-load it.

Lookup order (first match wins)
-------------------------------
1. **Environment variable** `PSYMATCH_MODEL_PATH`
   Absolute or relative path to a LightGBM booster file (`.txt`, `.json`, …).
2. **MLflow Model Registry**
   Queries for the newest model named `psymatch_ranker` in the *Production* stage.

If neither yields a valid file, returns `None`. Logs warnings on failures.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
from structlog import get_logger

log = get_logger()

# ---------------------------------------------------------------------- #
# 1. Environment variable lookup
# ---------------------------------------------------------------------- #
def _from_env() -> Optional[str]:
    """Use PSYMATCH_MODEL_PATH if it exists on the local file-system."""
    p = os.getenv("PSYMATCH_MODEL_PATH")
    if not p:
        return None
    path = Path(p).expanduser()
    if path.is_file():
        log.info("Model path from env var", path=str(path))
        return str(path)
    log.warning("Env var PSYMATCH_MODEL_PATH set but file not found", path=str(path))
    return None

# ---------------------------------------------------------------------- #
# 2. MLflow Model Registry lookup
# ---------------------------------------------------------------------- #
def _from_mlflow() -> Optional[str]:
    """Return path of newest *Production* model `psymatch_ranker` in MLflow."""
    try:
        import mlflow  # type: ignore
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='psymatch_ranker'")
        prod = [v for v in versions if v.current_stage == "Production"]
        if not prod:
            log.info("No Production models found in MLflow registry")
            return None
        # sort by creation timestamp desc
        prod.sort(key=lambda v: v.creation_timestamp, reverse=True)
        source = prod[0].source
        src_path = Path(source)
        if src_path.is_file():
            log.info("Model path from MLflow registry", path=str(src_path))
            return str(src_path)
        # if it's a directory, look for common filenames
        for fname in ["model.txt", "model.json", "model.bin"]:
            candidate = src_path / fname
            if candidate.is_file():
                log.info("Model file in MLflow artefact dir", path=str(candidate))
                return str(candidate)
        log.warning("MLflow source is directory but no model file found", dir=str(src_path))
    except Exception as exc:
        log.warning("MLflow lookup failed", error=str(exc))
    return None

# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #
def get_latest_production_model() -> Optional[str]:
    """
    Return a local file path to the newest LambdaRank booster, or `None`
    if no model is available.
    """
    # 1. Env var override
    path = _from_env()
    if path:
        return path
    # 2. MLflow registry fallback
    return _from_mlflow()
