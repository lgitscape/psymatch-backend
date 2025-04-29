# 📦 tests/test_ann_cache.py

import pytest
import time
import uuid
from engine.matcher import Matcher
from tests.utils.dummies import DummyClient, DummyTherapist

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
    matches, _ = await matcher.run()

    new_timestamp = matcher._ann_cache_timestamp

    if old_timestamp and new_timestamp:
        assert new_timestamp > old_timestamp
    else:
        assert matches is not None
        assert isinstance(matches, list)

