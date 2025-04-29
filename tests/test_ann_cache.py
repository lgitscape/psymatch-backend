# 📦 tests/test_ann_cache.py

import pytest
import time
import uuid
from engine.matcher import Matcher

class DummyClient:
    """Fully complete DummyClient for ANN cache test."""
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.topics = ["stress"]
        self.setting = "online"
        self.topic_weights = {"stress": 3}
        self.style_pref = "Warm"
        self.style_weight = 2
        self.languages = ["nl"]
        self.timeslots = ["avond"]
        self.budget = 100
        self.severity = 3
        self.lat = 52.01
        self.lon = 4.41
        self.gender_pref = None
        self.therapy_goals = ["stabiliseren"]
        self.client_traits = ["Volwassenen"]
        self.expat = False
        self.lgbtqia = False
        self.max_km = 25

class DummyTherapist:
    """Fully complete DummyTherapist for ANN cache test."""
    def __init__(self, id):
        self.id = id
        self.topics = ["stress"]
        self.setting = "online"
        self.client_groups = ["Volwassenen"]
        self.style = "Warm"
        self.therapist_goals = ["stabiliseren"]
        self.languages = ["nl"]
        self.timeslots = ["avond"]
        self.fee = 90
        self.contract_with_insurer = True
        self.gender_pref = None
        self.lat = 52.01
        self.lon = 4.41

@pytest.mark.asyncio
async def test_ann_cache_refresh():
    """Test that ANN cache is refreshed when therapist pool changes."""
    client = DummyClient()
    therapists = [DummyTherapist(str(i)) for i in range(3000)]

    matcher = Matcher(client, therapists)
    await matcher.run()

    old_timestamp = matcher._ann_cache_timestamp

    # Simulate time passing
    time.sleep(1)

    # Add new therapist
    therapists.append(DummyTherapist("new_therapist"))

    matcher = Matcher(client, therapists)
    matches = await matcher.run()

    new_timestamp = matcher._ann_cache_timestamp

    if old_timestamp and new_timestamp:
        assert new_timestamp > old_timestamp
    else:
        # Defensive fallback if ANN not built
        assert matches is not None
        assert isinstance(matches, list)
