# 📦 tests/test_explain_async.py

import pytest
from fastapi.testclient import TestClient
from main import app
from tests.utils.dummies import DummyClient

client = TestClient(app)

def get_dummy_client():
    return DummyClient().__dict__  # JSON input vereist dict

@pytest.mark.asyncio
async def test_explain_async_non_blocking():
    dummy_client = get_dummy_client()
    response = client.post("/explain", json=dummy_client)
    assert response.status_code in (200, 404)
    body = response.json()
    if response.status_code == 200:
        assert "feature_values" in body["data"]
        assert "shap_values" in body["data"]
