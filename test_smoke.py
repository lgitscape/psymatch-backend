"""tests/test_smoke.py – expanded smoke tests for PsyMatch duo-algorithm"""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient
import recommender
from recommender import ClientProfile, TherapistProfile, _recommend, settings

# Create a FastAPI test client
default_client = TestClient(recommender.app)

@pytest.fixture(scope="module")
def dummy_client() -> ClientProfile:
    return ClientProfile(
        setting="online",
        max_km=50,
        topics=["stress", "depressie"],
        topic_weights={"stress": 2, "depressie": 1},
        style_pref="warm",
        style_weight=4,
        gender_pref="Geen voorkeur",
        therapy_goals=["stabiliseren"],
        languages=["nl"],
        timeslots=["avond", "ochtend"],
        budget=100.0,
        severity=3,
        lat=52.0,
        lon=4.0,
    )

@pytest.fixture(scope="module")
def dummy_therapists() -> list[TherapistProfile]:
    base = dict(
        setting="online",
        topics=["stress"],
        style="warm",
        therapist_goals=["stabiliseren"],
        modalities=["stress"],
        client_groups=["Volwassenen"],
        languages=["nl"],
        timeslots=["avond"],
        fee=90.0,
        contract_with_insurer=False,
        gender_pref="Vrouw",
        lat=52.1,
        lon=4.1,
    )
    # Create variations on base
    th_style = TherapistProfile(id="th_style", **{**base, 'style': 'direct'})
    th_language = TherapistProfile(id="th_lang", **{**base, 'languages': ['en']})
    th_timeslot = TherapistProfile(id="th_time", **{**base, 'timeslots': ['middag']})
    th_inbudget = TherapistProfile(id="th_budget1", **{**base, 'fee': 80.0})
    th_slight_over = TherapistProfile(id="th_budget2", **{**base, 'fee': 115.0})
    th_far_over = TherapistProfile(id="th_budget3", **{**base, 'fee': 200.0})
    th_near = TherapistProfile(id="th_near", **{**base, 'lat': 52.0, 'lon': 4.0})
    th_far = TherapistProfile(id="th_far", **{**base, 'lat': 53.0, 'lon': 5.0})
    return [th_style, th_language, th_timeslot,
            th_inbudget, th_slight_over, th_far_over,
            th_near, th_far]


def test_rule_pipeline(dummy_client, dummy_therapists):
    res = _recommend(dummy_client, dummy_therapists, top_n=10)
    assert isinstance(res, list)
    assert len(res) <= len(dummy_therapists)


def test_model_pipeline(monkeypatch, dummy_client, dummy_therapists):
    # Force no model available
    if recommender.model_registry_helper:
        monkeypatch.setattr(
            recommender.model_registry_helper,
            "get_latest_production_model",
            lambda: None,
        )
    settings.model_path = None

    res = _recommend(dummy_client, dummy_therapists, top_n=3)
    assert isinstance(res, list)
    assert len(res) <= 3


def test_api_endpoints(monkeypatch):
    # /recommend empty
    response = default_client.post("/recommend", json={
        "setting": "online", "max_km": 0,
        "topics": [], "topic_weights": {},
        "style_pref": "warm", "style_weight": 1,
        "languages": ["nl"], "timeslots": ["avond"],
    })
    assert response.status_code == 200
    assert response.json() == []

    # /explain no shap
    monkeypatch.setattr(recommender, 'shap', None)
    resp = default_client.post("/explain", json={
        "setting": "online", "max_km": 0,
        "topics": [], "topic_weights": {},
        "style_pref": "warm", "style_weight": 1,
        "languages": ["nl"], "timeslots": ["avond"],
    })
    assert resp.status_code == 501

    # /model/reload no model
    settings.model_path = None
    if recommender.model_registry_helper:
        monkeypatch.setattr(
            recommender.model_registry_helper,
            'get_latest_production_model', lambda: None
        )
    reload_resp = default_client.post("/model/reload")
    assert reload_resp.status_code == 500

    # /match endpoints no celery
    monkeypatch.setattr(recommender, 'celery_app', None)
    r1 = default_client.post("/match/1/start", params={"client_email": "a@b.com"})
    r2 = default_client.post("/match/1/end", params={"client_email": "a@b.com"})
    assert r1.status_code == 501
    assert r2.status_code == 501


def test_style_preference(dummy_client, dummy_therapists):
    # Ensure style match influences ranking
    res = _recommend(dummy_client, dummy_therapists, top_n=2)
    # First result should have style 'warm'
    assert res[0][0].style == dummy_client.style_pref


def test_language_and_timeslot_overlap(dummy_client, dummy_therapists):
    # Therapists speaking 'nl' and with 'avond' timeslot should outrank others
    res = _recommend(dummy_client, dummy_therapists, top_n=2)
    first = res[0][0]
    assert 'nl' in first.languages
    assert 'avond' in first.timeslots


def test_fee_logic(dummy_client, dummy_therapists):
    # in-budget > slightly over > far over
    ids = [t.id for t, _ in _recommend(dummy_client, dummy_therapists, top_n=3)]
    assert ids.index('th_budget1') < ids.index('th_budget2') < ids.index('th_budget3')


def test_distance_filtering(dummy_client, dummy_therapists):
    # th_far should be dropped by max_km
    filtered = [t.id for t, _ in _recommend(dummy_client, dummy_therapists, top_n=10)]
    assert 'th_near' in filtered
    assert 'th_far' not in filtered


def test_severity_weight(dummy_client):
    # Higher severity yields slightly higher score
    cli_low = dummy_client.copy(update={"severity": 1})
    cli_high = dummy_client.copy(update={"severity": 5})
    th = TherapistProfile(
        id="th_s", setting="online", topics=["stress"], style="warm",
        therapist_goals=["stabiliseren"], modalities=["stress"], client_groups=["Volwassenen"],
        languages=["nl"], timeslots=["avond"], fee=100.0, contract_with_insurer=False,
        gender_pref=None, lat=52.0, lon=4.0
    )
    low_score = _recommend(cli_low, [th], top_n=1)[0][1]
    high_score = _recommend(cli_high, [th], top_n=1)[0][1]
    assert high_score > low_score


def test_client_group_ok(dummy_client):
    # If client.client_groups matches therapist.client_groups, client_group_ok=1
    cli = dummy_client.copy(update={"client_groups": ["Volwassenen"]})
    th_ok = TherapistProfile(
        id="th_cg_ok", setting="online", topics=["stress"], style="warm",
        therapist_goals=["stabiliseren"], modalities=["stress"], client_groups=["Volwassenen"],
        languages=["nl"], timeslots=["avond"], fee=90.0,
        contract_with_insurer=False, gender_pref=None, lat=52.0, lon=4.0
    )
    th_no = TherapistProfile(
        id="th_cg_no", setting="online", topics=["stress"], style="warm",
        therapist_goals=["stabiliseren"], modalities=["stress"], client_groups=["Jongeren"],
        languages=["nl"], timeslots=["avond"], fee=90.0,
        contract_with_insurer=False, gender_pref=None, lat=52.0, lon=4.0
    )
    ids = [t.id for t, _ in _recommend(cli, [th_no, th_ok], top_n=2)]
    assert ids[0] == "th_cg_ok"

def test_modality_overlap(dummy_client):
    # Client prefers ACT; therapist offers ACT vs. no overlap
    client = dummy_client.copy(update={"preferred_modalities": ["ACT"]})
    th_yes = TherapistProfile(..., modalities=["ACT"], ...)
    th_no  = TherapistProfile(..., modalities=["EMDR"], ...)
    ids = [t.id for t,_ in _recommend(client, [th_no, th_yes], top_n=2)]
    assert ids[0] == th_yes.id

