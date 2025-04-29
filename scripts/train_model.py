# scripts/train_model.py
# Final production-quality version: Train a universal LightGBM LambdaRank model with Optuna hyperparameter tuning

import pandas as pd
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_optuna
import os
from pathlib import Path
from typing import Tuple
from engine.features import build_feature_vector
from supabase_client import supabase
import structlog
from datetime import datetime
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

log = structlog.get_logger()

# Settings
MODEL_DIR = Path("models/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def fetch_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        clients_resp = supabase.table("test_clients").select("*").execute()
        therapists_resp = supabase.table("test_therapists").select("*").execute()
        matches_resp = supabase.table("test_training_data").select("*").execute()

        clients = pd.DataFrame(clients_resp.data)
        therapists = pd.DataFrame(therapists_resp.data)
        matches = pd.DataFrame(matches_resp.data)

        return clients, therapists, matches
    except Exception as e:
        log.error("Failed to fetch data", error=str(e))
        raise

def build_training_data(matches: pd.DataFrame, clients: pd.DataFrame, therapists: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    feature_rows = []
    labels = []
    groups = []

    id2client = {row["id"]: row for _, row in clients.iterrows()}
    id2therapist = {row["id"]: row for _, row in therapists.iterrows()}

    for client_id, group in matches.groupby("client_id"):
        client_data = id2client.get(client_id)
        if client_data is None:
            continue
        group_size = 0

        for _, match in group.iterrows():
            therapist_data = id2therapist.get(match["therapist_id"])
            if therapist_data is None:
                continue

            c = SimpleNamespace(**client_data)
            t = SimpleNamespace(**therapist_data)

            fv = build_feature_vector(c, t)

            initial_score = match.get("initial_score", 5) / 10.0
            fv["initial_score"] = initial_score

            feature_rows.append(list(fv.values()))

            final_score = match.get("final_score")
            if final_score is not None:
                labels.append(float(final_score) / 10.0)
                group_size += 1
            else:
                log.warning("Skipping match with missing final_score", client_id=client_id)

        if group_size > 0:
            groups.append(group_size)

    return pd.DataFrame(feature_rows), labels, groups

def train_model(X_train: pd.DataFrame, y_train: list, groups: list) -> None:
    log.info("Starting model training (with hyperparameter tuning)", samples=len(y_train), groups=len(groups))

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    lgb_train = lgb.Dataset(X_tr, label=y_tr, group=groups)
    lgb_valid = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "ndcg_eval_at": [5, 10],
        "random_state": 42,
    }

    log.info("Starting Optuna hyperparameter tuning...")

    tuner = lgb_optuna.LightGBMTuner(
        params,
        lgb_train,
        valid_sets=[lgb_valid],
        num_boost_round=500,
        early_stopping_rounds=20,
        verbose_eval=50,
        time_budget=600,
        optuna_seed=42,
    )

    tuner.run()

    best_params = tuner.best_params
    log.info(f"Best hyperparameters found: {best_params}")

    best_params.update({
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "verbosity": -1,
        "random_state": 42,
    })

    model = lgb.train(
        best_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(20),
            lgb.log_evaluation(50)
        ]
    )

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = MODEL_DIR / f"model_finalscore_{timestamp}.txt"
    model.save_model(str(model_path))

    log.info("Model saved", path=str(model_path))
    log.info(f"Best iteration: {model.best_iteration}, Best valid NDCG@10: {model.best_score['valid']['ndcg@10']:.4f}")

def main() -> None:
    clients, therapists, matches = fetch_data()
    X_train, y_train, groups = build_training_data(matches, clients, therapists)
    if not X_train.empty and y_train:
        train_model(X_train, y_train, groups)
    else:
        log.error("No valid training data found.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Training pipeline failed", error=str(e))
        raise
