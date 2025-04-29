# PsyMatch Matching Engine

Een schaalbaar, transparant en privacy-bewust matchingsysteem voor cliënten en therapeuten.

![Coverage](./coverage.svg)

## Visie van PsyMatch

PsyMatch gelooft dat een goede match tussen cliënt en therapeut verder gaat dan alleen praktische zaken. We streven naar:

- **Inhoudelijke afstemming**: gedeelde thema’s, doelen en werkstijlen.
- **Praktische haalbaarheid**: locatie, beschikbaarheid, taal en budget.
- **Persoonlijke voorkeuren**: voorkeur voor gender, setting (online/fysiek), doelgroep.
- **Zelfbeschikking**: het algoritme adviseert, de cliënt kiest.
- **Uitlegbaarheid**: elke match is transparant en herleidbaar.
- **Privacy**: geen opslag of logging van persoonsgegevens.

## Belangrijkste functies

- Modulaire pipeline: filters → features → matcher → shortlist
- ANN-index voor snelle shortlist bij grote aantallen therapeuten
- LambdaRank fallback scoring + SHAP-compatibele uitleg
- Metrics via Prometheus + lifecyclebeheer voor caching
- Scoreweging configureerbaar via `weights.yml`
- Volledige inputvalidatie, geen logging van PII

## Installatie

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Voorbeeldgebruik

```bash
python main.py --client examples/client_a.json
```

## CLI-opties

```bash
python main.py --client pad/naar/client.json [--rebuild-ann]
```

## Ontwikkelaarscommando’s (via Makefile)

```bash
make test          # Testen uitvoeren
make run           # Matchingsengine uitvoeren
make rebuild-ann   # ANN-index handmatig heropbouwen
```

## Match-output (voorbeeld)

Elke match bestaat uit een gescoorde shortlist met uitleg per component. Bij gebruik van SHAP is een aparte uitleg op te vragen via het `/explain` endpoint.

```json
{
  "client_id": "c123",
  "top_matches": [
    {
      "therapist_id": "t456",
      "score": 91.2,
      "explanation": {
        "topic_overlap": 0.8,
        "style_match": 1.0,
        "distance_km": 5.1
      }
    }, ...
  ]
}
```

## Schematisch overzicht

![Systeemdiagram](docs/system_diagram.svg)

## Documentatie

- [Architectuur](docs/architecture.md)
- [API-routes](docs/api.md)
- [Privacybeleid](docs/privacy.md)

## Wijzigingen

Zie `CHANGELOG.md` voor de versiegeschiedenis.

## Licentie

© PsyMatch — All rights reserved.
