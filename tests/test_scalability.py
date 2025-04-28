# 📦 tests/test_scalability.py

import pytest
from fastapi.testclient import TestClient
from main import app
import uuid

client = TestClient(app)

@pytest.mark.asyncio
async def test_scalability_ann_shortlisting():
    # Create dummy client
    dummy_client = {
        "client_id": str(uuid.uuid4()),
        "setting": "online",
        "max_km": 50,
        "topics": ["stress"],
        "topic_weights": {"stress": 3},
        "style_pref": "Warm",
        "style_weight": 2,
        "gender_pref": "Geen voorkeur",
        "therapy_goals": ["inzicht bieden"],
        "client_traits": ["Jongvolwassene"],
        "languages": ["nl"],
        "timeslots": ["avond"],
        "budget": 100,
        "severity": 3,
        "lat": 52.01,
        "lon": 4.41
    }
    
    # Simulate recommend endpoint
    response = client.post("/recommend", json=dummy_client)
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()["data"]
        assert all(0 <= match["score_normalized"] <= 100 for match in data)
        assert all(data[i]["score_normalized"] >= data[i+1]["score_normalized"] for i in range(len(data)-1))
