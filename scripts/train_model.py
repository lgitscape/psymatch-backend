import pandas as pd
import lightgbm as lgb
import os
from pathlib import Path
from typing import Tuple, Dict
from engine.features import build_feature_vector
from supabase_client import supabase
import structlog
from datetime import datetime
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error
from types import SimpleNamespace
import optuna
import numpy as np
import joblib

log = structlog.get_logger()

# Settings
MODEL_DIR = Path("models/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
TEST_SIZE = 0.1


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

        for _, match in group.iterrows():
            therapist_data = id2therapist.get(match["therapist_id"])
            if therapist_data is None:
                continue

            c = SimpleNamespace(**client_data)
            t = SimpleNamespace(**therapist_data)

            fv = build_feature_vector(c, t)

            initial_score = match.get("initial_score", 5) / 10.0
            fv["initial_score"] = initial_score

            feature_rows.append({**fv, "client_id": client_id})

            final_score = match.get("final_score")
            if final_score is not None:
                labels.append(float(final_score) / 10.0)
            else:
                log.warning("Skipping match with missing final_score", client_id=client_id)

    df = pd.DataFrame(feature_rows)
    client_ids = df.pop("client_id")
    return df, labels, client_ids

def train_model(X: pd.DataFrame, y: list, client_ids: list, n_trials: int = 100) -> None:
    log.info("Starting model training", samples=len(y))

    X = X.reset_index(drop=True)
    y = np.array(y)
    client_ids = np.array(client_ids)

    # Split into train/test before any training
    X_train, X_test, y_train, y_test, client_ids_train, _ = train_test_split(
        X, y, client_ids, test_size=TEST_SIZE, random_state=42, stratify=None
    )

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 512),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        }

        gkf = GroupKFold(n_splits=N_SPLITS)
        rmses = []

        for train_idx, val_idx in gkf.split(X_train, y_train, groups=client_ids_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                valid_names=['valid'],
                num_boost_round=10000,
                early_stopping_rounds=100,
                verbose_eval=100
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            rmse = mean_squared_error(y_val, preds, squared=False)
            rmses.append(rmse)

        return np.mean(rmses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    log.info(f"Best trial: RMSE = {study.best_value:.5f}")
    log.info(f"Best params: {study.best_params}")

    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42
    })

    # Train N_SPLITS models for ensembling
    models = []
    gkf = GroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=client_ids_train)):
        X_tr, _ = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, _ = y_train[train_idx], y_train[val_idx]

        dtrain_full = lgb.Dataset(X_tr, label=y_tr)
        model = lgb.train(
            best_params,
            dtrain_full,
            num_boost_round=1000
        )

        models.append(model)

        model_path = MODEL_DIR / f"model_fold{fold}.txt"
        model.save_model(str(model_path))
        log.info(f"Saved fold model {fold}", path=str(model_path))

    # Evaluate ensemble on test set
    preds = np.mean([model.predict(X_test) for model in models], axis=0)
    test_rmse = mean_squared_error(y_test, preds, squared=False)
    log.info(f"Test RMSE (ensemble): {test_rmse:.5f}")

    # Save Optuna study
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    study_path = MODEL_DIR / f"optuna_study_{timestamp}.pkl"
    joblib.dump(study, study_path)

    log.info("Training complete.")

def main() -> None:
    clients, therapists, matches = fetch_data()
    X_train, y_train, client_ids = build_training_data(matches, clients, therapists)
    if not X_train.empty and y_train:
        train_model(X_train, y_train, client_ids)
    else:
        log.error("No valid training data found.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Training pipeline failed", error=str(e))
        raise
