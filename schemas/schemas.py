# 📦 /schemas/schemas.py

from pydantic import BaseModel
from typing import List, Optional
import uuid

class ClientProfile(BaseModel):
    client_id: uuid.UUID
    setting: str
    max_km: int
    topics: List[str]
    topic_weights: dict[str, int]
    style_pref: str
    style_weight: int
    gender_pref: Optional[str] = None
    therapy_goals: List[str] = []
    client_traits: List[str] = []
    languages: List[str]
    timeslots: List[str]
    budget: Optional[float] = None
    severity: int = 3
    lat: Optional[float] = None
    lon: Optional[float] = None

class MatchResult(BaseModel):
    therapist_id: str
    score_raw: float
    score_normalized: float

class RecommendResponse(BaseModel):
    status: str
    data: List[MatchResult]

class ExplainResponse(BaseModel):
    status: str
    data: dict

class ChooseMatchResponse(BaseModel):
    status: str
    message: str
    match_id: str
    therapist_id: str

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    version: str

class ErrorResponse(BaseModel):
    status: str
    message: str
    info: Optional[str | dict] = None
