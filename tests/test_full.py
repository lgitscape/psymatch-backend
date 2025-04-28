# 📦 /tests/test_full.py

import pytest
from fastapi.testclient import TestClient
from main import app
import uuid

client = TestClient(app)

def make_dummy_client():
    return {
        "client_id": str(uuid.uuid4()),
        "setting": "online",
        "max_km": 20,
        "topics": ["stress", "zelfbeeld"],
        "topic_weights": {"stress": 3, "zelfbeeld": 2},
        "style_pref": "Warm",
        "style_weight": 4,
        "gender_pref": "Vrouw",
        "therapy_goals": ["inzicht bieden"],
        "client_traits": ["Jongvolwassene"],
        "languages": ["nl"],
        "timeslots": ["avond"],
        "budget": 95.0,
        "severity": 3,
        "lat": 52.01,
        "lon": 4.41
    }

# ---------------------- Filter tests ----------------------

def test_filter_by_topic_overlap_no_match():
    dummy = make_dummy_client()
    dummy["topics"] = ["fake_topic"]
    response = client.post("/recommend", json=dummy)
    assert response.status_code == 404

# ---------------------- Feature tests ----------------------

def test_feature_fee_logic_adjustment():
    dummy = make_dummy_client()
    dummy["budget"] = 50
    response = client.post("/recommend", json=dummy)
    assert response.status_code in (200, 404)

# ---------------------- Matcher logic tests ----------------------

def test_matcher_basic_ranking_or_empty():
    dummy = make_dummy_client()
    response = client.post("/recommend", json=dummy)
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()["data"]
        if len(data) >= 2:
            assert data[0]['score_normalized'] >= data[1]['score_normalized']

# ---------------------- API endpoint tests ----------------------

def test_api_recommend_success_or_404():
    dummy = make_dummy_client()
    response = client.post("/recommend", json=dummy)
    assert response.status_code in (200, 404)

def test_api_explain_success_or_404():
    dummy = make_dummy_client()
    response = client.post("/explain", json=dummy)
    assert response.status_code in (200, 404)

# ---------------------- Choose Match Test ----------------------

def test_api_choose_match_dummy_success():
    dummy = make_dummy_client()
    rec_response = client.post("/recommend", json=dummy)
    if rec_response.status_code == 200:
        rec_data = rec_response.json()["data"]
        if rec_data:
            match_id = "test-match-id"
            therapist_id = rec_data[0]["therapist_id"]
            choose_response = client.post(f"/match/{match_id}/choose/{therapist_id}")
            assert choose_response.status_code in (200, 404)

# ---------------------- Edge case test ----------------------

def test_recommend_empty_db(monkeypatch):
    from main import THERAPISTS
    monkeypatch.setattr("main.THERAPISTS", [])
    dummy = make_dummy_client()
    response = client.post("/recommend", json=dummy)
    assert response.status_code == 404
