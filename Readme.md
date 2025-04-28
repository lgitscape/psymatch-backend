PsyMatch Matching Engine

Production-grade AI matching system for therapist-client compatibility.

![Coverage](./coverage.svg)

Overview
--------

PsyMatch is a production-grade, scalable, resilient matching engine built to optimize therapist-client matching based on style, topics, availability, distance, and insurance considerations.
The system uses progressive fallback filtering, ANN (Approximate Nearest Neighbors) shortlisting, stylistic diversity promotion, and explainable AI (SHAP) to ensure transparency and fairness.

Features
--------

- Progressive fallback logic (distance -> budget -> topic)
- ANN-based scaling for more than 30,000 therapists
- Async explainability API (SHAP with top-feature impact)
- LambdaRank model ranking fallback
- Full Prometheus metrics for monitoring
- Future-proof retraining endpoint architecture

Running Tests
-------------

Run the following command to execute all tests and generate coverage reports:

pytest --cov

This will generate a coverage.svg badge and an HTML coverage report in the htmlcov/ directory.

Project Structure
-----------------

/engine: Core matching logic, features, models
/api: FastAPI routes and handlers
/services: Service layer
/utils: Utility helpers (Supabase client, retries)
/tests: Unit and integration tests

Installation
------------

pip install -r requirements.txt
pip install -r requirements-dev.txt

License
-------

© PsyMatch — All rights reserved.
