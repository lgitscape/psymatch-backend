# 📦 /tests/test_smoke.py

import uuid
from fastapi.testclient import TestClient
from main import app
import pytest
import asyncio
from tests.utils.dummies import DummyClient

client = TestClient(app)

pytest_plugins = ("pytest_asyncio",)

def get_dummy_client():
    return DummyClient().__dict__  # voor API-test met JSON input

@pytest.mark.asyncio
async def test_matcher_direct_async():
    from engine.matcher import Matcher
    client = DummyClient()
    matcher = Matcher(client, [])
    results, _ = await matcher.run()
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
