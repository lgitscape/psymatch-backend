# 📦 engine/matcher.py
# ─────────────────────────────
# Full matching engine for PsyMatch

from engine import filters, features
import structlog
from engine.models import lambda_model
import pandas as pd
from collections import defaultdict
from pathlib import Path
import yaml
import copy

config_path = Path(__file__).resolve().parent.parent / "config" / "weights.yml"
with open(config_path, "r") as f:
    CONFIG_WEIGHTS = yaml.safe_load(f)

log = structlog.get_logger()

MINIMUM_NORMALIZED_SCORE = 80.0  # 80% match threshold

class Matcher:
    def __init__(self, client, therapists, weights=None, ann_helper=None):
        self.client = client
        self.therapists = therapists
        self.ann_helper = ann_helper
        self.weights = weights or self._select_weights()
        self._ann_cache_vecs = None
        self._ann_cache_ids = None
        self._ann_cache_timestamp = None

    def _select_weights(self):
        """Choose appropriate weights profile based on client attributes."""
        # For now, always default
        return CONFIG_WEIGHTS.get("default")

        """In the future, weights will be chosen along the following code example:"""
        # if self.client.severity >= 5:
        #     return CONFIG_WEIGHTS.get("profile_high_severity", CONFIG_WEIGHTS["default"])
        # elif self.client.budget and self.client.budget < 80:
        #     return CONFIG_WEIGHTS.get("profile_cost_sensitive", CONFIG_WEIGHTS["default"])

    async def run(self, top_n=10):
        candidates = self._apply_filters()

        if not candidates:
            log.warning("No therapists passed filters for client", client_id=str(self.client.client_id))
            return [], "none"

        if self.ann_helper and len(candidates) > 2000:
            import time
            if (self._ann_cache_vecs is None
                or time.time() - (self._ann_cache_timestamp or 0) > 300
                or len(self._ann_cache_ids) != len(candidates)):

                # Rebuild ANN index every 5 minutes
                feat_df = features.build_feature_vector(self.client, candidates)
                vecs = feat_df.drop(columns=["th_idx"]).astype("float32").to_numpy()
                ids = feat_df["th_idx"].tolist()
        
                ann = self.ann_helper.ANNIndex(vecs.shape[1])
                ann.build(vecs, ids)
        
                self._ann_cache_vecs = ann
                self._ann_cache_ids = {id: th for id, th in zip(ids, candidates)}
                self._ann_cache_timestamp = time.time()
        
                log.info("Built ANN cache", candidates=len(candidates))

            query_vec = features.build_feature_vector(self.client, candidates).drop(columns=["th_idx"]).mean().to_numpy()
            top_ids = self._ann_cache_vecs.query(query_vec, k=200)
            candidates = [self._ann_cache_ids[i] for i in top_ids if i in self._ann_cache_ids]

            log.info("Applied ANN shortlist", remaining=len(candidates))

        features_list = []
        ids_list = []
        style_dict = {}

        for th in candidates:
            fv = self._normalize_feature_vector(features.build_feature_vector(self.client, th))
            features_list.append(fv)
            ids_list.append(th.id)
            style_dict[th.id] = th.style

        feat_df = pd.DataFrame(features_list)

        used_algorithm = "rule"
        try:
            if lambda_model and hasattr(lambda_model, "models") and lambda_model.models:
                scores = lambda_model.predict(feat_df)
                log.info("Scored matches using Regression ensemble model")
                used_algorithm = "regression_ensemble"

            else:
                scores = [self._calculate_score(fv) for fv in features_list]
                log.info("Scored matches using rule-based engine")
        except Exception as e:
            log.error("Model prediction failed, falling back to rule engine", error=str(e))
            FALLBACK_COUNTER.inc()
            scores = [self._calculate_score(fv) for fv in features_list]

        scored = list(zip(ids_list, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        diversified = self._diversify_matches(scored, style_dict, top_n)

        max_score = self._max_possible_score()

        normalized = [
            {
                "therapist_id": th_id,
                "score_raw": round(score, 2),
                "score_normalized": round((score / max_score) * 100, 2),
            }
            for th_id, score in diversified
        ]

        # ➔ Enforce minimum match threshold
        filtered_normalized = [
            match for match in normalized if match["score_normalized"] >= MINIMUM_NORMALIZED_SCORE
        ]

        self._log_client_group_distribution(candidates)
        log.info("Top matches generated", client_id=str(self.client.client_id), matches=filtered_normalized)

        return filtered_normalized, used_algorithm

    def _normalize_feature_vector(self, feat_dict):
        """Apply feature normalization so no single feature dominates."""
        normalized = {}
        for k, v in feat_dict.items():
            if k == "distance_km":
                # Transform distance to similarity (e.g., 0 km = 1.0, 50 km = 0.0 cutoff)
                if v <= 5:
                    normalized[k] = 1.0
                elif v <= 20:
                    normalized[k] = 1.0 - ((v - 5) / 15)
                else:
                    normalized[k] = 0.0
            elif k == "severity":
                normalized[k] = v / 5.0  # Severity [1-5] → [0.2–1.0]
            else:
                normalized[k] = v  # assume all others [0,1]
        return normalized

    def _diversify_matches(self, scored, style_dict, top_n):
        """Promote stylistic diversity directly after scoring."""
        seen_styles = set()
        diversified = []
        for th_id, score in scored:
            style = style_dict.get(th_id)
            if style not in seen_styles or len(diversified) < top_n // 2:
                diversified.append((th_id, score))
                seen_styles.add(style)
            if len(diversified) >= top_n:
                break
        return diversified

    def _apply_filters(self):
        """Apply hard filters with progressive relaxation."""
        filtered = [th for th in self.therapists if filters.apply_all_filters(self.client, th)]
        if filtered:
            return filtered
    
        log.warning("No therapists after strict filters, relaxing travel distance...", client_id=str(self.client.client_id))
        relaxed_client = copy.deepcopy(self.client)
        relaxed_client.max_km += 10
    
        filtered = [th for th in self.therapists if filters.apply_all_filters(relaxed_client, th)]
        if filtered:
            return filtered
    
        log.warning("No therapists after relaxed distance, now relaxing budget constraint...", client_id=str(self.client.client_id))
        # Relax budget margin (+30 EUR soft)
        def relaxed_apply_all_filters(cli, th):
            return (
                filters.filter_by_setting(cli, th)
                and filters.filter_by_language(cli, th)
                and filters.filter_by_distance(cli, th)
                and filters.filter_by_topic_overlap(cli, th)
                and (th.contract_with_insurer or th.fee <= (cli.budget + 30 if cli.budget else th.fee))
            )
    
        filtered = [th for th in self.therapists if relaxed_apply_all_filters(relaxed_client, th)]
        if filtered:
            return filtered
    
        log.warning("No therapists after travel and budget relaxation, now relaxing topic overlap requirement...", client_id=str(self.client.client_id))
        def most_relaxed_filter(cli, th):
            return (
                filters.filter_by_setting(cli, th)
                and filters.filter_by_language(cli, th)
                and filters.filter_by_distance(cli, th)
            )

        filtered = [th for th in self.therapists if most_relaxed_filter(relaxed_client, th)]
        return filtered

    def _calculate_score(self, feat_dict):
        return sum(feat_dict.get(k, 0.0) * self.weights.get(k, 0.0) for k in feat_dict)

    def _max_possible_score(self):
        return sum(abs(w) for w in self.weights.values() if w > 0)

    def _log_client_group_distribution(self, therapists):
        """Log distribution of client_groups among therapists."""
        import collections
        counter = collections.Counter()
        for th in therapists:
            if th.client_groups:
                for cg in th.client_groups:
                    counter[cg] += 1
        log.info("Therapist client group distribution", distribution=dict(counter))

async def predict_for_client(client, therapists, top_n=50):
    """
    Predict top N therapist matches for a given client.
    Returns list of matches sorted by predicted score (highest first).
    """
    matcher = Matcher(client, therapists)
    matches, algorithm = await matcher.run(top_n=top_n)

    return {
        "matches": matches,
        "algorithm": algorithm
    }
