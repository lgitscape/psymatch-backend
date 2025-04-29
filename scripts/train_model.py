import pandas as pd
import lightgbm as lgb
import os
from pathlib import Path
from typing import Tuple, Dict, List
from engine.features import build_feature_vector
from supabase_client import supabase
import structlog
from datetime import datetime
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import root_mean_squared_error
from types import SimpleNamespace
import optuna
import numpy as np
import joblib

# Globale statusvariabele
training_status = {"status": "idle", "progress": 0}

log = structlog.get_logger()

# Settings
MODEL_DIR = Path("models/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
TEST_SIZE = 0.1
CATEGORICAL_COLUMNS = ["item_category_1", "item_category_2", "item_category_3"]  # Example categorical features


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

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df, labels, client_ids


def train_model(
    X: pd.DataFrame,
    y: List[float],
    client_ids: List[str],
    n_trials: int = 5,
    save_models: bool = True,
    show_progress: bool = True
) -> None:
    global training_status

    log.info("Starting model training", samples=len(y))

    X = X.reset_index(drop=True)
    y = np.array(y)
    client_ids = np.array(client_ids)

    X_train, X_test, y_train, y_test, client_ids_train, _ = train_test_split(
        X, y, client_ids, test_size=TEST_SIZE, random_state=42
    )

    categorical_features = [col for col in CATEGORICAL_COLUMNS if col in X_train.columns]

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

            dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_features)
            dvalid = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                valid_names=['valid'],
                num_boost_round=10000,
                callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            rmse = root_mean_squared_error(y_val, preds)
            rmses.append(rmse)

        return np.mean(rmses)

    study = optuna.create_study(direction="minimize")

    if show_progress:
        total_trials = n_trials
        trial_counter = 0

        for _ in range(n_trials):
            study.optimize(objective, n_trials=1)
            trial_counter += 1
            progress = (trial_counter / total_trials) * 100
            training_status["progress"] = int(progress)
            log.info(f"Training progress: {int(progress)}%")
    else:
        study.optimize(objective, n_trials=n_trials)

    log.info("Best trial", rmse=study.best_value, params=study.best_params)

    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42
    })

    # Train ensemble
    models = []
    gkf = GroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=client_ids_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]

        dtrain_full = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_features)

        model = lgb.train(
            best_params,
            dtrain_full,
            num_boost_round=1000
        )

        models.append(model)

    if save_models:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        models_path = MODEL_DIR / f"ensemble_models_{timestamp}.pkl"
        joblib.dump(models, models_path)
        log.info("Saved ensemble models", path=str(models_path))

    preds = np.mean([model.predict(X_test) for model in models], axis=0)
    test_rmse = root_mean_squared_error(y_test, preds)
    log.info("Test RMSE (ensemble)", test_rmse=test_rmse)

    study_path = MODEL_DIR / f"optuna_study_{timestamp}.pkl"
    joblib.dump(study, study_path)

    log.info("Training complete.")


def predict_ensemble(models: List[lgb.Booster], X_new: pd.DataFrame) -> np.ndarray:
    preds = np.mean([model.predict(X_new, num_iteration=model.best_iteration) for model in models], axis=0)
    return preds


def load_models(models_path: Path) -> List[lgb.Booster]:
    return joblib.load(models_path)


def main(n_trials: int = 100) -> None:
    global training_status
    try:
        training_status["status"] = "running"
        training_status["progress"] = 0
        clients, therapists, matches = fetch_data()
        X_train, y_train, client_ids = build_training_data(matches, clients, therapists)

        if not X_train.empty and y_train:
            train_model(X_train, y_train, client_ids, n_trials=n_trials)
            training_status["status"] = "completed"
            training_status["progress"] = 100
        else:
            log.error("No valid training data found.")
            training_status["status"] = "failed"
    except Exception as e:
        training_status["status"] = "failed"
        training_status["progress"] = 0
        log.error("Training pipeline failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
