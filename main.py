# psyMatch – Production Recommender Core (v5.4.2)
# =================================================
"""Full-feature duo-algorithm service with extended questionnaire coverage (v5.4.2).
Patched to include client preferred modalities and client groups."""

from __future__ import annotations

import os
import json
import logging
import uuid
from datetime import datetime
from math import radians, sin, cos, atan2, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from fastapi import FastAPI, BackgroundTasks, HTTPException
from lightgbm import LGBMRanker  # type: ignore
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from structlog import get_logger

# Optional dependencies ----------------------------------------------------
try:
    import ann_helper  # provides ANNIndex + shortlist()
except ModuleNotFoundError:
    ann_helper = None  # type: ignore

try:
    import model_registry_helper  # provides get_latest_production_model()
except ModuleNotFoundError:
    model_registry_helper = None  # type: ignore

try:
    from celery import Celery  # noqa: N811
except ModuleNotFoundError:
    Celery = None  # type: ignore

try:
    import shap  # type: ignore
except ModuleNotFoundError:
    shap = None  # type: ignore

# ───────────────────────── 0. Settings & metrics ──────────────────────────
class Settings(BaseSettings):
    fairness_eps: float = 0.05
    model_path: Optional[str] = None  # override model location
    redis_host: str = "localhost"     # Celery broker/backend

    class Config:
        env_file = ".env"

settings = Settings()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = get_logger()

REQ = Counter("requests_total", "API calls", ["ep"])
LAT = Histogram("latency_sec", "latency", ["ep"])
DP_GAP = Gauge("dem_par_gap", "Demographic-parity gap", ["attr"])

# ───────────────────────── 1. Domain models ───────────────────────────────
class ClientProfile(BaseModel):
    client_id: uuid.UUID                     # nodig voor logging
    setting: str                             # online | fysiek | beide | geen
    max_km: int                              # max travel distance (km)
    topics: List[str]                        # up to 3 topics
    topic_weights: Dict[str, int]
    style_pref: str                          # Praktisch | Warm | Direct | Reflectief | Geen voorkeur
    style_weight: int                        # 1–5
    gender_pref: Optional[str] = None        # Geen voorkeur | Vrouw | Man | Non-binair
    therapy_goals: List[str] = []            # up to 3 goals
    preferred_modalities: List[str] = []     # gewenste therapievormen
    client_traits: List[str] = []  	     # e.g. ["Jongvolwassene", "LGBTQIA+", "Expat"]
    languages: List[str]
    timeslots: List[str]                     # Ochtend | Middag | Avond
    budget: Optional[float] = None           # per session (€)
    severity: int = 3                        # Likert scale 1–5
    lat: Optional[float] = None
    lon: Optional[float] = None

class TherapistProfile(BaseModel):
    id: str
    setting: str                            # online | fysiek | beide
    modalities: List[str] = []              # CGT, ACT, EMDR, …
    topics: List[str]
    client_groups: List[str] = []           # Volwassenen, Jongvolwassenen, …
    style: str                              # Praktisch | Warm | Direct | Reflectief
    therapist_goals: List[str] = []         # Mensen in beweging, Stabiliseren, …
    languages: List[str]
    timeslots: List[str]
    fee: float                              # per session (€)
    contract_with_insurer: bool
    gender_pref: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

# ───────────────────────── 2. Feature engineering ─────────────────────────
RULE_W: Dict[str, float] = {
    "weighted_topic_overlap": 2.0,
    "style_match": 1.0,
    "style_weight": 0.5,
    "language_overlap": 1.0,
    "timeslot_overlap": 0.5,
    "fee_score": 1.0,
    "budget_penalty": -1.0,
    "gender_ok": 0.5,
    "goal_overlap": 1.5,
    "modality_overlap": 1.0,
    "client_group_ok": 1.0,
    "severity": 0.2,
    "distance_km": -0.01,
}

def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    R = 6371.0
    φ1, φ2 = map(radians, (a_lat, b_lat))
    dφ, dλ = radians(b_lat - a_lat), radians(b_lon - a_lon)
    a = sin(dφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(dλ / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def _build_features(cli: ClientProfile, ths: List[TherapistProfile]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    # precompute client preferences
    c_topics = set(cli.topics)
    c_goals = set(cli.therapy_goals)
    c_mods = set(cli.preferred_modalities)
    c_traits = set(cli.client_traits)
    total = sum(cli.topic_weights.values()) or 1

    for th in ths:
        # filter by setting
        if cli.setting == "online" and th.setting not in {"online", "beide"}:
            continue
        if cli.setting == "fysiek" and th.setting not in {"fysiek", "beide"}:
            continue
        # distance filter
        dist = 0.0
        if cli.setting != "online" and cli.lat is not None and th.lat is not None:
            dist = _haversine_km(cli.lat, cli.lon, th.lat, th.lon)
            if dist > cli.max_km:
                continue
        # weighted topic overlap
        wto = sum(cli.topic_weights.get(t, 1) for t in c_topics & set(th.topics)) / total
        # fee logic
        fee_rel = cli.budget is not None and not th.contract_with_insurer
        fee_delta = (cli.budget - th.fee) if fee_rel else 0.0
        fee_score = 1.0 if fee_rel and fee_delta >= 0 else (0.5 if fee_rel and fee_delta >= -20 else 0.0)
        budget_penalty = -1.0 if fee_rel and fee_delta < 0 else 0.0
        # gender preference
        gender_ok = 1.0 if (cli.gender_pref in (None, "Geen voorkeur") or cli.gender_pref == th.gender_pref) else 0.0
        # therapy goal overlap
        goal_overlap = float(len(c_goals & set(th.therapist_goals)) / max(1, len(c_goals)))
        # modality overlap
        modality_overlap = float(bool(c_mods & set(th.modalities)))
        # client group match
        client_group_ok = float(bool(c_traits & set(th.client_groups))) if c_traits else 1.0

        rows.append({
            "th_idx": th.id,
            "weighted_topic_overlap": wto,
            "style_match": float(cli.style_pref == th.style),
            "style_weight": float(cli.style_weight),
            "language_overlap": float(bool(set(cli.languages) & set(th.languages))),
            "timeslot_overlap": float(len(set(cli.timeslots) & set(th.timeslots))),
            "fee_score": fee_score,
            "fee_delta": fee_delta,
            "budget_penalty": budget_penalty,
            "gender_ok": gender_ok,
            "goal_overlap": goal_overlap,
            "modality_overlap": modality_overlap,
            "client_group_ok": client_group_ok,
            "severity": float(cli.severity),
            "distance_km": dist,
        })
    return pd.DataFrame(rows)

# ───────────────────────── 3. Scorers ─────────────────────────────────────
def _rule_scores(cli: ClientProfile, df: pd.DataFrame) -> np.ndarray:
    base = sum(df[col] * w for col, w in RULE_W.items())
    return base.to_numpy()

class RankerModel:
    def __init__(self, booster_path: Optional[str] = None):
        self.model = LGBMRanker(objective="lambdarank", metric="ndcg")
        if booster_path:
            self.model.booster_.load_model(booster_path)
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

_model: Optional[RankerModel] = None

def _load_model(force: bool = False) -> Optional[RankerModel]:
    global _model
    if _model and not force:
        return _model
    path = settings.model_path or (model_registry_helper.get_latest_production_model() if model_registry_helper else None)
    if path:
        _model = RankerModel(path)
        log.info("LambdaRank model loaded", path=path)
    else:
        log.warning("No model found → rule engine active")
        _model = None
    return _model

# ───────────────────────── 4. Fairness gauge ─────────────────────────────
def _update_dp(recs: List[Tuple[TherapistProfile, float]]) -> None:
    cnt: Dict[str, int] = {}
    for th, _ in recs:
        g = getattr(th, "gender_pref", "unk") or "unk"
        cnt[g] = cnt.get(g, 0) + 1
    if len(cnt) >= 2:
        v = list(cnt.values())
        DP_GAP.labels("gender_pref").set(abs(v[0] / sum(v) - v[1] / sum(v)))

# ───────────────────────── 5. Recommendation pipeline ──────────────────
def _recommend(cli: ClientProfile, ths: List[TherapistProfile], top_n: int = 10) -> List[Tuple[TherapistProfile, float]]:
    # ANN shortlist
    if ann_helper and len(ths) > 2000:
        ths = ann_helper.shortlist(cli, ths, k=200, feature_fn=_build_features)

    feat = _build_features(cli, ths)
    if feat.empty:
        return []

    mdl = _load_model()
    scores = _rule_scores(cli, feat) if mdl is None else mdl.predict(feat.drop(columns=["th_idx"]))
    feat["score"] = scores
    ranked = feat.sort_values("score", ascending=False)
    id2 = {t.id: t for t in ths}
    res = [(id2[r.th_idx], r.score) for r in ranked.itertuples()][:top_n]
    _update_dp(res)
    log_match_recommendation(cli, ths, res, feat)
    return res

from supabase_client import supabase  # jouw init script

def log_match_recommendation(cli: ClientProfile, ths: List[TherapistProfile], res: List[Tuple[TherapistProfile, float]], features_df: pd.DataFrame):
    if not res:
       return
    match_data = [{
        "therapist_id": t.id,
        "score": float(s),
        "rank": i + 1
    } for i, (t, s) in enumerate(res)]

    row = {
        "client_id": cli.client_id,  # voeg toe aan ClientProfile!
        "algorithm": "rule",  # of "lambdarank-v1"
        "recommended": match_data,
	"feature_vector": features_df.drop(columns=["th_idx"]).iloc[0].dropna().to_dict(),
    }
    supabase.table("match_logs").insert(row).execute()

# ───────────────────────── 6. Celery SRS ─────────────────────────────────
if Celery:
    celery_app = Celery(
        "srs",
        broker=f"redis://{settings.redis_host}:6379/0",
        backend=f"redis://{settings.redis_host}:6379/1",
    )

    @celery_app.task(bind=True, autoretry_for=(Exception,), max_retries=3, retry_backoff=True)
    def send_srs_email(self, email: str, link: str) -> None:
        import smtplib
        with smtplib.SMTP("localhost") as s:
            s.sendmail("no-reply@psymatch.nl", [email], f"""Subject: SRS

Klik: {link}""")
else:
    celery_app = None  # type: ignore

# ───────────────────────── 7. FastAPI app ────────────────────────────────
app = FastAPI(title="PsyMatch Recommender v5.4.2")

THERAPISTS: List[TherapistProfile] = [
    TherapistProfile(
        id="th1",
        setting="online",
        modalities=["ACT"],
        topics=["stress", "zelfbeeld"],
        client_groups=["Jongvolwassene", "Expat"],
        style="Warm",
        therapist_goals=["inzicht bieden", "zelfbeeld versterken"],
        languages=["nl"],
        timeslots=["avond"],
        fee=90.0,
        contract_with_insurer=False,
        gender_pref="Vrouw",
        lat=52.01,
        lon=4.41
    ),
    TherapistProfile(
        id="th2",
        setting="online",
        modalities=["CGT"],
        topics=["stress", "angst"],
        client_groups=["Volwassenen"],
        style="Direct",
        therapist_goals=["mensen in beweging brengen"],
        languages=["nl", "en"],
        timeslots=["avond", "ochtend"],
        fee=100.0,
        contract_with_insurer=False,
        gender_pref="Geen voorkeur",
        lat=52.05,
        lon=4.5
    )
]

@app.on_event("startup")
async def _startup() -> None:
    _load_model()
    start_http_server(8001)

@app.post("/recommend")
async def recommend_ep(client: ClientProfile, top_n: int = 10):
    tic = datetime.utcnow()
    REQ.labels("recommend").inc()
    results = _recommend(client, THERAPISTS, top_n)
    LAT.labels("recommend").observe((datetime.utcnow() - tic).total_seconds())
    return [{"id": t.id, "score": round(s, 4)} for t, s in results]

@app.post("/explain")
async def explain_ep(client: ClientProfile):
    if shap is None:
        raise HTTPException(status_code=501, detail="SHAP not installed")
    stub = TherapistProfile(
        id="stub", setting="online", topics=client.topics,
        style=client.style_pref, languages=client.languages,
        timeslots=client.timeslots, fee=0.0,
        contract_with_insurer=True
    )
    feat = _build_features(client, [stub])
    explainer = shap.Explainer(lambda X: _rule_scores(client, pd.DataFrame(X)), feat)
    sv = explainer(feat)
    return {
        "base_value": float(sv.base_values[0]),
        "feature_values": feat.iloc[0].to_dict(),
        "shap_values": dict(zip(feat.columns, sv.values[0].tolist())),
    }

@app.post("/model/reload")
async def reload_ep():
    if _load_model(force=True) is None:
        raise HTTPException(status_code=500, detail="No model found")
    return {"status": "reloaded"}

@app.post("/match/{mid}/start")
async def match_start_ep(mid: int, client_email: str, bg: BackgroundTasks):
    if celery_app is None:
        raise HTTPException(status_code=501, detail="Celery not configured")
    token = str(uuid.uuid4())
    link = f"https://psymatch.nl/srs/{token}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
