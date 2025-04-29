# scripts/setup_training_data.py

import random
import uuid
import time

from supabase_client import supabase
from engine.features import build_feature_vector

# ───────────────────────────────────────────────────────────────────────────
# Stap A: 50 fake therapists genereren en uploaden naar test_therapists
# ───────────────────────────────────────────────────────────────────────────
def generate_fake_therapists(n=50):
    topics_pool    = ["stress", "depressie", "angst", "relatie", "verlies"]
    styles         = ["Warm", "Direct", "Reflectief", "Praktisch"]
    timeslots_pool = ["ochtend", "middag", "avond"]
    genders        = ["Vrouw", "Man", None]
    setting        = ["Fysiek", "Online", "Geen voorkeur"]

    therapists = []
    for _ in range(n):
        th = {
            "id": str(uuid.uuid4()),
            "setting": random.choice(setting),
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

    print(f"Uploaden van {len(therapists)} therapists naar test_therapists…")
    resp = supabase.table("test_therapists").insert(therapists).execute()
    print("=== RESP DEBUG ===", flush=True)
    for attr in dir(resp):
        if not attr.startswith("_"):
            try:
                value = getattr(resp, attr)
                print(f"{attr}: {value}", flush=True)
            except Exception as e:
                print(f"{attr}: <error getting value> {e}", flush=True)
    if resp.raw_response.status_code >= 400:
        raise RuntimeError(f"Failed to insert therapists: {resp.data}")
    print("Therapists succesvol geüpload.")

    return therapists

# ───────────────────────────────────────────────────────────────────────────
# Stap B: 50 fake clients genereren en uploaden naar test_clients
# ───────────────────────────────────────────────────────────────────────────
def generate_fake_clients(n=50):
    topics_pool    = ["stress", "depressie", "angst", "relatie", "verlies"]
    styles         = ["Warm", "Direct", "Reflectief", "Praktisch"]
    timeslots_pool = ["ochtend", "middag", "avond"]
    client_groups_pool = ["Kinderen", "Adolescenten", "Volwassenen", "Ouderen"]
    languages_pool = ["nl", "en"]
    genders        = ["Man", "Vrouw", None]
    setting        = ["Fysiek", "Online", "Geen voorkeur"]
    expat_status   = [True, False]
    lgbtqia_status = [True, False]

    clients = []
    for _ in range(n):
        cl = {
            "id": str(uuid.uuid4()),
            "setting": random.choice(setting),
            "max_km": random.choice([10, 25, 50]),
            "topics": random.sample(topics_pool, k=random.randint(1, 2)),
            "topic_weights": {topic: random.randint(1, 3) for topic in random.sample(topics_pool, k=random.randint(1, 2))},
            "style_pref": random.choice(styles),
            "style_weight": random.randint(1, 3),
            "gender_pref": random.choice(genders),
            "therapy_goals": ["stabiliseren"],
            "client_traits": random.sample(client_groups_pool, k=1),
            "languages": random.sample(languages_pool, k=1),
            "timeslots": random.sample(timeslots_pool, k=random.randint(1, 2)),
            "budget": round(random.uniform(60, 120), 2),
            "severity": random.randint(1, 5),
            "lat": 52.0 + random.uniform(-0.5, 0.5),
            "lon": 4.0 + random.uniform(-0.5, 0.5),
            "expat": random.choice(expat_status),
            "lgbtqia": random.choice(lgbtqia_status),
        }
        clients.append(cl)

    print(f"Uploaden van {len(clients)} clients naar test_clients…")
    resp = supabase.table("test_clients").insert(clients).execute()
    if resp.raw_response.status_code >= 400:
        raise RuntimeError(f"Failed to insert clients: {resp.data}")
    print("Clients succesvol geüpload.")

    return clients

# ───────────────────────────────────────────────────────────────────────────
# Stap C: 2500 matches genereren tussen clients en therapists
# ───────────────────────────────────────────────────────────────────────────
def generate_and_upload_matches(clients, therapists, n_matches=2500):
    records = []

    for _ in range(n_matches):
        client = random.choice(clients)

        # Bepaal overlap score met alle therapists
        therapist_scores = []
        for therapist in therapists:
            overlap = len(set(client["topics"]) & set(therapist["topics"]))
            therapist_scores.append((therapist, overlap))

        # Sorteer therapists op hoogste topic-overlap
        therapist_scores.sort(key=lambda x: x[1], reverse=True)

        # Kies random uit top 5 beste matches (of minder als er weinig overlap is)
        top_choices = [th for th, score in therapist_scores[:5] if score > 0]
        if not top_choices:
            # Als geen overlap, fallback naar random therapist
            top_choices = [therapist for therapist, _ in therapist_scores]

        # Simuleer keuze met kansverdeling
        choice_weights = [0.7, 0.2, 0.07, 0.02, 0.01]  # optelling = 1
        idx = random.choices(range(len(top_choices)), weights=choice_weights[:len(top_choices)])[0]
        therapist = top_choices[idx]

        # Dummy Client en Therapist objecten voor build_feature_vector
        class C:
            client_id     = client["id"]
            setting       = client["setting"]
            max_km        = client["max_km"]
            topics        = client["topics"]
            topic_weights = client["topic_weights"]
            style_pref    = client["style_pref"]
            style_weight  = client["style_weight"]
            gender_pref   = client["gender_pref"]
            therapy_goals = client["therapy_goals"]
            client_traits = client["client_traits"]
            languages     = client["languages"]
            timeslots     = client["timeslots"]
            budget        = client["budget"]
            severity      = client["severity"]
            lat           = client["lat"]
            lon           = client["lon"]

        class T:
            id                   = therapist["id"]
            setting              = therapist["setting"]
            topics               = therapist["topics"]
            client_groups        = therapist["client_groups"]
            style                = therapist["style"]
            therapist_goals      = therapist["therapist_goals"]
            languages            = therapist["languages"]
            timeslots            = therapist["timeslots"]
            fee                  = therapist["fee"]
            contract_with_insurer= therapist["contract_with_insurer"]
            gender_pref          = therapist["gender_pref"]
            lat                  = therapist["lat"]
            lon                  = therapist["lon"]

        fv = build_feature_vector(C, T)

        # Realistische label (0, 1 of 2)
        rel = (fv["weighted_topic_overlap"] * 2 + fv["style_match"] + fv["language_overlap"]) / 4
        label = int(rel * 3)

        # Simuleer initial en final score
        initial_score = max(1, min(10, round(rel * 4 + random.uniform(0, 2))))  # Base + random noise
        final_score   = max(1, min(10, round(rel * 4 + random.uniform(-1, 2))))  # Possibly slightly better/worse
        
        fv["client_id"]     = client["id"]
        fv["therapist_id"]  = therapist["id"]
        fv["label"]         = label
        fv["initial_score"] = initial_score
        fv["final_score"]   = final_score
        fv["rank_in_top"]  = idx + 1  # 1 = beste match, 2 = tweede, etc.

        records.append(fv)


    print(f"Uploaden van {len(records)} match records naar test_training_data…")
    chunk_size = 500
    for i in range(0, len(records), chunk_size):
        batch = records[i : i + chunk_size]
        resp  = supabase.table("test_training_data").insert(batch).execute()
        if resp.raw_response.status_code >= 400:
            raise RuntimeError(f"Failed to insert training_data batch: {resp.data}")
        time.sleep(0.5)
    print("Match records succesvol geüpload.")


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
def main():
    therapists = generate_fake_therapists(50)
    clients    = generate_fake_clients(50)
    generate_and_upload_matches(clients, therapists, 2500)
    print("Klaar: 50 therapists, 50 clients en 2500 match records staan in Supabase.")


if __name__ == "__main__":
    main()
