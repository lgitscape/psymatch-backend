# scripts/setup_training_data.py

import random
import uuid
import time

import pandas as pd
from supabase_client import supabase
from engine.features import build_feature_vector

# ───────────────────────────────────────────────────────────────────────────
# Stap A: 50 fake therapists genereren en uploaden
# ───────────────────────────────────────────────────────────────────────────
def generate_fake_therapists(n=50):
    topics_pool    = ["stress", "depressie", "angst", "relatie", "verlies"]
    styles         = ["Warm", "Direct", "Reflectief", "Praktisch"]
    timeslots_pool = ["ochtend", "middag", "avond"]
    genders        = ["Vrouw", "Man", None]

    therapists = []
    for _ in range(n):
        th = {
            "id": str(uuid.uuid4()),
            "setting": "online",
            "topics": random.sample(topics_pool, k=random.randint(1, 3)),
            "client_groups": ["Volwassenen"],
            "style": random.choice(styles),
            "therapist_goals": ["stabiliseren"],
            "languages": ["nl"],
            "timeslots": random.sample(timeslots_pool, k=random.randint(1, 2)),
            "fee": round(random.uniform(50, 150), 2),
            "contract_with_insurer": random.choice([True, False]),
            "gender_pref": random.choice(genders),
            "lat": 52.0 + random.uniform(-0.5, 0.5),
            "lon": 4.0  + random.uniform(-0.5, 0.5),
        }
        therapists.append(th)

    print(f"Uploaden van {len(therapists)} therapists naar Supabase…")
    resp = supabase.table("therapists").insert(therapists).execute()
    if resp.status_code >= 400:
        raise RuntimeError(f"Failed to insert therapists: {resp.data}")
    print("Therapists succesvol geüpload.")

    return [th["id"] for th in therapists]

# ───────────────────────────────────────────────────────────────────────────
# Stap B: Dummy client definiëren
# ───────────────────────────────────────────────────────────────────────────
def make_dummy_client():
    class Client:
        client_id = uuid.uuid4()
        setting = "online"
        max_km = 50
        topics = ["stress", "angst"]
        topic_weights = {"stress": 2, "angst": 1}
        style_pref = "Warm"
        style_weight = 3
        gender_pref = None
        therapy_goals = ["stabiliseren"]
        client_traits = ["Volwassenen"]
        languages = ["nl"]
        timeslots = ["ochtend", "avond"]
        budget = 100.0
        severity = 3
        lat = 52.0
        lon = 4.0
    return Client()

# ───────────────────────────────────────────────────────────────────────────
# Stap C: 2000 match-records genereren en uploaden
# ───────────────────────────────────────────────────────────────────────────
def generate_and_upload_matches(client, therapist_ids, n_matches=2000):
    records = []
    # Ophalen volledige therapist objecten uit Supabase
    thr_resp = supabase.table("therapists").select("*").execute()
    all_ths = thr_resp.data

    # Build map id→therapist voor feature vector
    id2th = {th["id"]: th for th in all_ths}

    for _ in range(n_matches):
        th_id = random.choice(therapist_ids)
        th = id2th[th_id]

        # maak een dummy Therapist-object zoals engine/features verwacht
        class T:
            id = th["id"]
            setting = th["setting"]
            topics = th["topics"]
            client_groups = th["client_groups"]
            style = th["style"]
            therapist_goals = th["therapist_goals"]
            languages = th["languages"]
            timeslots = th["timeslots"]
            fee = th["fee"]
            contract_with_insurer = th["contract_with_insurer"]
            gender_pref = th["gender_pref"]
            lat = th["lat"]
            lon = th["lon"]

        fv = build_feature_vector(client, T)
        # label realistisch: combinatie van topic-overlap, style, language
        rel = (fv["weighted_topic_overlap"] * 2 + fv["style_match"] + fv["language_overlap"]) / 4
        label = int(rel * 3)

        # Voeg client_id en therapist_id toe voor traceerbaarheid
        fv["client_id"]      = str(client.client_id)
        fv["therapist_id"]   = th_id
        fv["label"]          = label
        records.append(fv)

    print(f"Uploaden van {len(records)} match records naar Supabase…")
    # Chunked insert om timeouts te vermijden
    chunk_size = 500
    for i in range(0, len(records), chunk_size):
        batch = records[i : i + chunk_size]
        resp = supabase.table("training_data").insert(batch).execute()
        if resp.status_code >= 400:
            raise RuntimeError(f"Failed to insert training_data batch: {resp.data}")
        time.sleep(0.5)
    print("Match records succesvol geüpload.")

# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
def main():
    # Stap A: therapists
    therapist_ids = generate_fake_therapists(50)

    # Stap B: dummy client
    client = make_dummy_client()

    # Stap C: matches
    generate_and_upload_matches(client, therapist_ids, 2000)

    print("Klaar: 50 therapists en 2000 match records staan in Supabase.")

if __name__ == "__main__":
    main()
