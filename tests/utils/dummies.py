import uuid

class DummyClient:
    """Volledig gevulde client met attributen."""
    def __init__(self):
        self.client_id = str(uuid.uuid4())
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
        self.therapy_goals = ["stabiliseren"]
        self.client_traits = ["Volwassenen"]
        self.expat = False
        self.lgbtqia = False
        self.max_km = 25

class DummyTherapist:
    def __init__(self, id):
        self.id = id
        self.topics = ["stress"]
        self.setting = "online"
        self.client_groups = ["Volwassenen"]
        self.style = "Warm"
        self.therapist_goals = ["stabiliseren"]
        self.languages = ["nl"]
        self.timeslots = ["avond"]
        self.fee = 90
        self.contract_with_insurer = True
        self.gender_pref = None
        self.lat = 52.01
        self.lon = 4.41
