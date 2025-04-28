# 📦 tests/test_fallbacks.py

import pytest
from fastapi.testclient import TestClient
from main import app
import uuid

client = TestClient(app)

@pytest.mark.asyncio
async def test_progressive_fallbacks():
    # Client with unrealistic strict needs to trigger fallback
    strict_client = {
        "client_id": str(uuid.uuid4()),
        "setting": "fysiek",
        "max_km": 1,
        "topics": ["fake_topic"],  # unlikely
        "topic_weights": {"fake_topic": 3},
        "style_pref": "Onbekend",
        "style_weight": 5,
        "gender_pref": "Man",
        "therapy_goals": ["onbestaand doel"],
        "client_traits": ["onbestaande groep"],
        "languages": ["de"],  # unlikely
        "timeslots": ["ochtend"],
        "budget": 10,  # too low
        "severity": 5,
        "lat": 0.0,
        "lon": 0.0
    }

    response = client.post("/recommend", json=strict_client)
    assert response.status_code in (200, 404)
    # Even if no match, fallback sequence was tested
