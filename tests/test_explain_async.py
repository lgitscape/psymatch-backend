# 📦 tests/test_explain_async.py

import pytest
from fastapi.testclient import TestClient
from main import app
import uuid

client = TestClient(app)

@pytest.mark.asyncio
async def test_explain_async_non_blocking():
    dummy_client = {
        "client_id": str(uuid.uuid4()),
        "setting": "online",
        "max_km": 20,
        "topics": ["stress", "zelfbeeld"],
        "topic_weights": {"stress": 3, "zelfbeeld": 2},
        "style_pref": "Warm",
        "style_weight": 4,
        "gender_pref": "Geen voorkeur",
    }
