# 📦 engine/filters.py
# ─────────────────────────────
# All hard filters for PsyMatch v5.5.0

from math import radians, sin, cos, sqrt, atan2

def filter_by_setting(cli, th):
    """Client setting preference (online/fysiek/beide)."""
    return (cli.setting == "online" and th.setting in {"online", "beide"}) or \
           (cli.setting == "fysiek" and th.setting in {"fysiek", "beide"}) or \
           (cli.setting == "geen" or cli.setting == "beide")

def filter_by_language(cli, th):
    """At least 1 shared language."""
    return bool(set(cli.languages) & set(th.languages))

def filter_by_distance(cli, th):
    """Max travel distance."""
    if cli.setting == "online" or cli.lat is None or th.lat is None:
        return True
    return _haversine_km(cli.lat, cli.lon, th.lat, th.lon) <= cli.max_km

def filter_by_topic_overlap(cli, th):
    """At least 1 overlapping topic."""
    return bool(set(cli.topics) & set(th.topics))

def filter_by_budget(cli, th):
    """Budget hard cut-off if no contract with insurer."""
    if cli.budget is None:
        return True
    if th.contract_with_insurer:
        return True
    return th.fee <= cli.budget + 20  # +20 soft margin zoals besproken

def apply_all_filters(cli, th):
    """Applies all hard filters sequentially."""
    return (
        filter_by_setting(cli, th)
        and filter_by_language(cli, th)
        and filter_by_distance(cli, th)
        and filter_by_topic_overlap(cli, th)
        and filter_by_budget(cli, th)
    )

# ─────────────────────────────
# Internal helper

def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Calculate haversine distance between two lat/lon points."""
    R = 6371.0
    φ1, φ2 = map(radians, (a_lat, b_lat))
    dφ, dλ = radians(b_lat - a_lat), radians(b_lon - a_lon)
    a = sin(dφ / 2)**2 + cos(φ1) * cos(φ2) * sin(dλ / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
