# 📦 tests/test_ann_cache.py

import pytest
import time
from engine.matcher import Matcher

class DummyClient:
    def __init__(self):
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

class DummyTherapist:
    def __init__(self, id):
        self.id = id
        self.topics = ["stress"]
        self.setting = "online"
        self.languages = ["nl"]
        self.timeslots = ["avond"]
        self.fee = 90
        self.contract_with_insurer = True
        self.style = "Warm"
        self.therapist_goals = []
        self.client_groups = []
        self.lat = 52.01
        self.lon = 4.41

@pytest.mark.asyncio
async def test_ann_cache_refresh():
    client = DummyClient()
    therapists = [DummyTherapist(str(i)) for i in range(3000)]

    matcher = Matcher(client, therapists)
    await matcher.run()

    old_timestamp = matcher._ann_cache_timestamp

    # Simulate time passing
    time.sleep(1)

    # Add therapist → triggers cache refresh
    therapists
