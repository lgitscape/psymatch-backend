# 📦 engine/features.py
# ─────────────────────────────
# All feature vector calculations for PsyMatch v5.5.0

from math import radians, sin, cos, sqrt, atan2

def weighted_topic_overlap(cli, th):
    """Weighted overlap of topics between client and therapist."""
    c_topics = set(cli.topics)
    t_topics = set(th.topics)
    total_weight = sum(cli.topic_weights.values()) or 1
    return sum(cli.topic_weights.get(t, 1) for t in c_topics & t_topics) / total_weight

def style_match(cli, th):
    """Exact style match."""
    return 1.0 if cli.style_pref == th.style else 0.0

def style_weight(cli):
    """Weight assigned by client to style preference."""
    return float(cli.style_weight)

def language_overlap(cli, th):
    """Shared language(s) between client and therapist."""
    return float(bool(set(cli.languages) & set(th.languages)))

def timeslot_overlap(cli, th):
    """Shared available timeslots."""
    return float(len(set(cli.timeslots) & set(th.timeslots)))

def fee_score(cli, th):
    """Score based on fee compared to budget."""
    if cli.budget is None or th.contract_with_insurer:
        return 1.0
    if cli.budget >= th.fee:
        return 1.0
    elif cli.budget >= th.fee - 20:
        return 0.5
    else:
        return 0.0

def budget_penalty(cli, th):
    """Penalty for budget overflow."""
    if cli.budget is None or th.contract_with_insurer:
        return 0.0
    return -1.0 if cli.budget < th.fee else 0.0

def gender_ok(cli, th):
    """Gender preference match."""
    if cli.gender_pref in (None, "Geen voorkeur"):
        return 1.0
    return 1.0 if cli.gender_pref == th.gender_pref else 0.0

def goal_overlap(cli, th):
    """Shared therapy goals."""
    c_goals = set(cli.therapy_goals)
    t_goals = set(th.therapist_goals)
    return float(len(c_goals & t_goals) / max(1, len(c_goals)))

def client_group_ok(cli, th):
    """Shared client trait group."""
    if not cli.client_traits:
        return 1.0
    return float(bool(set(cli.client_traits) & set(th.client_groups)))

def severity(cli):
    """Severity of client's complaints (Likert 1-5)."""
    return float(cli.severity)

def distance_km(cli, th):
    """Distance between client and therapist in km."""
    if cli.lat is None or th.lat is None:
        return 0.0
    return _haversine_km(cli.lat, cli.lon, th.lat, th.lon)

# ─────────────────────────────
# Internal

def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Calculate haversine distance between two lat/lon points."""
    R = 6371.0
    φ1, φ2 = map(radians, (a_lat, b_lat))
    dφ, dλ = radians(b_lat - a_lat), radians(b_lon - a_lon)
    a = sin(dφ / 2)**2 + cos(φ1) * cos(φ2) * sin(dλ / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ─────────────────────────────
# Full feature vector builder

def build_feature_vector(cli, th):
    """Assemble full feature vector for a client-therapist pair."""
    raw_features = {
        "weighted_topic_overlap": weighted_topic_overlap(cli, th),
        "style_match": style_match(cli, th),
        "style_weight": style_weight(cli),
        "language_overlap": language_overlap(cli, th),
        "timeslot_overlap": timeslot_overlap(cli, th),
        "fee_score": fee_score(cli, th),
        "budget_penalty": budget_penalty(cli, th),
        "gender_ok": gender_ok(cli, th),
        "goal_overlap": goal_overlap(cli, th),
        "client_group_ok": client_group_ok(cli, th),
        "severity": severity(cli),
        "distance_km": distance_km(cli, th),
    }
    return {k: float(v) for k, v in raw_features.items()}