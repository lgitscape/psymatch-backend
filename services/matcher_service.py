# 📦 /services/matcher_service.py

from engine.matcher import Matcher
from supabase_client import supabase
from utils.supabase_utils import insert_with_retry
from engine.models import lambda_model
from prometheus_client import Counter

REQUEST_COUNTER = Counter("psymatch_requests_total", "Total /recommend requests made")
FALLBACK_COUNTER = Counter("psymatch_fallbacks_total", "Total number of fallbacks to rule-based scoring")
MATCHES_RETURNED_COUNTER = Counter("psymatch_matches_returned", "Number of matches returned per request")
FILTERED_OUT_COUNTER = Counter("psymatch_matches_filtered_out", "Number of matches filtered out under minimum score")
MODEL_USAGE_COUNTER = Counter("psymatch_model_usage", "Algorithm used (rule vs lambdarank)", ["algorithm"])

async def run_matcher(client, therapists, top_n=10):
    matcher = Matcher(client, therapists)
    matches, algorithm_used = await matcher.run(top_n=top_n)

    if algorithm_used:
        MODEL_USAGE_COUNTER.labels(algorithm_used).inc()

    if matches:
        MATCHES_RETURNED_COUNTER.inc(len(matches))
    return matches, algorithm_used

async def run_explanation(client, therapists):
    matcher = Matcher(client, therapists)
    matches, _ = await matcher.run(top_n=1)

    if not matches:
        return None

    return matches[0]
