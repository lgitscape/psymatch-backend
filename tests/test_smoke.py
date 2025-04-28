# 📦 /tests/test_smoke.py

import uuid
from fastapi.testclient import TestClient
from main import app
import pytest
import asyncio

client = TestClient(app)

pytest_plugins = ("pytest_asyncio",)

def get_dummy_client():
    """Creates a dummy client profile for testing."""
    return {
        "client_id": str(uuid.uuid4()),
        "setting": "online",
        "max_km": 20,
        "topics": ["stress", "zelfbeeld"],
        "topic_weights": {"stress": 3, "zelfbeeld": 2},
        "style_pref": "Warm",
        "style_weight": 4,
        "gender_pref": "Geen voorkeur",
        "therapy_goals": ["inzicht bieden"],
        "client_traits": ["Jongvolwassene"],
        "languages": ["nl"],
        "timeslots": ["avond"],
        "budget": 95.0,
        "severity": 3,
        "lat": 52.01,
        "lon": 4.41
    }

@pytest.mark.asyncio
async def test_matcher_direct_async():
    """Directly test matcher logic."""
    from engine.matcher import Matcher
    dummy_client = get_dummy_client()
    matcher = Matcher(dummy_client, [])
    results, _ = await matcher.run(top_n=10)
    assert isinstance(results, list)

def test_recommend_success_or_404():
    """Test /recommend endpoint."""
    response = client.post("/recommend", json=get_dummy_client())
    assert response.status_code in (200, 404)
    body = response.json()
    if response.status_code == 200:
        assert body["status"] == "success"
        assert isinstance(body["data"], list)
        if body["data"]:
            assert "therapist_id" in body["data"][0]
            assert "score_normalized" in body["data"][0]
            assert 0 <= body["data"][0]["score_normalized"] <= 100
    if response.status_code == 404:
        assert body["status"] == "error"
        assert "80%" in body["message"]

def test_explain_success_or_404():
    """Test /explain endpoint."""
    response = client.post("/explain", json=get_dummy_client())
    assert response.status_code in (200, 404)
    body = response.json()
    if response.status_code == 200:
        assert body["status"] == "success"
        assert "feature_values" in body["data"]
        assert "shap_values" in body["data"]
