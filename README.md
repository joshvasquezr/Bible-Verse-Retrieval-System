# Bible Verse Retrieval System — Milestone 1

Simple theme classifier for Bible verses running in Docker. The API exposes a `/predict` endpoint that returns one of ten themes:

Love, Joy, Peace, Grief, Faith, Fear, Mercy, Grace, Doubt, Forgiveness

## Project structure

```
.
├── 00_init.sql
├── api
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── app
│   ├── Dockerfile
│   └── static
│       └── index.html
├── data
│   ├── get_data.py
│   └── kjv.txt
├── docker-compose.yaml
└── .env
```

## What’s included

- FastAPI service with two endpoints:
	- GET `/health` → status and count of KJV-derived stopwords
	- POST `/predict` → `{ theme, confidence }` for an input verse
- ML pipeline: TF‑IDF vectorizer + Multinomial Naive Bayes
- Light preprocessing: derive frequent/archaic tokens from `data/kjv.txt` as custom stopwords (falls back if file is missing)
- Minimal static app (Nginx) to poke the API
- Postgres (pgvector) service scaffolded for future milestones

## Prerequisites

- Docker and Docker Compose
- `.env` file with:

```
POSTGRES_USER=user
POSTGRES_PASSWORD=bible
POSTGRES_DB=bible_db
```

## Run it

```
docker compose up --build -d
```

API endpoints (host ports):
- API: http://localhost:5001
- App: http://localhost:3001

Quick checks:

```
curl -s http://localhost:5001/health
curl -s -X POST http://localhost:5001/predict \
	-H 'Content-Type: application/json' \
	-d '{"verse":"The Lord is my shepherd, I shall not want."}'
```

## Data usage

- `data/get_data.py` downloads the KJV text to `data/kjv.txt`.
- The API reads `data/kjv.txt` (mounted read-only) at startup to derive a small list of stopwords (e.g., archaic words like “thee”, “thou”, frequent function words). If the file isn’t available, the service runs without extra stopwords.
- For labeled training, place an optional `api/verses_themes.csv` with columns `verse,theme`. If absent, the API uses a tiny built‑in seed set per theme for Milestone 1.

## Development notes

- Host→container port mappings:
	- API: 5001 → 5000
	- App: 3001 → 3000
- Modify the seed data or add `verses_themes.csv` to improve results.
- Logs:
	- `docker compose logs -f api`
	- `docker compose logs -f app`

## Next stpes

- Expand labeled dataset and evaluation
- Persist embeddings/vectors in Postgres (pgvector)
- Add retrieval/search endpoints and frontend integration

