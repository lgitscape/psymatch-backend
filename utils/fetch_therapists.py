# 📦 utils/fetch_therapists.py

import asyncio
import structlog
from typing import List
from supabase_client import supabase
import json

log = structlog.get_logger()

class TherapistProfile:
    """Therapist profile model for internal matching use, with field validation."""
    def __init__(self, id, setting, topics, client_groups, style, therapist_goals, languages, timeslots, fee, contract_with_insurer, gender_pref=None, lat=None, lon=None):
        self.id = id
        self.setting = setting
        self.topics = self._safe("topics", topics)
        self.client_groups = self._safe("client_groups", client_groups)
        self.style = style
        self.therapist_goals = self._safe("therapist_goals", therapist_goals)
        self.languages = self._safe("languages", languages)
        self.timeslots = self._safe("timeslots", timeslots)
        self.fee = fee
        self.contract_with_insurer = contract_with_insurer
        self.gender_pref = gender_pref
        self.lat = lat
        self.lon = lon

    def _safe(self, fieldname, val):
        if isinstance(val, list):
            return val
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
            log.warning("Field parsed but is not list", field=fieldname, value=val)
            return []
        except Exception:
            log.warning("Invalid list field format", field=fieldname, value=val)
            return []

async def fetch_therapists(retries: int = 3, delay: float = 2.0) -> List[TherapistProfile]:
    """Fetch therapists from Supabase, with retry logic."""
    for attempt in range(retries):
        try:
            log.info(f"Fetching therapists (attempt {attempt+1})")
            response = supabase.table("test_therapists").select("*").execute()

            if not response.data:
                log.warning("No therapists found in Supabase.")
                return []

            therapists = []
            for th in response.data:
                therapists.append(TherapistProfile(
                    id=th.get("id"),
                    setting=th.get("setting"),
                    topics=th.get("topics"),
                    client_groups=th.get("client_groups"),
                    style=th.get("style"),
                    therapist_goals=th.get("therapist_goals"),
                    languages=th.get("languages"),
                    timeslots=th.get("timeslots"),
                    fee=th.get("fee"),
                    contract_with_insurer=th.get("contract_with_insurer"),
                    gender_pref=th.get("gender_pref"),
                    lat=th.get("lat"),
                    lon=th.get("lon"),
                ))
            log.info(f"Successfully fetched {len(therapists)} therapists from Supabase.")
            return therapists

        except Exception as e:
            log.error(f"Failed to fetch therapists (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                raise Exception("Startup failed: Could not fetch therapists from Supabase.") from e
