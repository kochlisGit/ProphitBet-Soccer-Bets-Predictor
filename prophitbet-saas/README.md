# ProphitBet SaaS Platform

AI-powered soccer predictions SaaS platform with freemium monetization. Built on top of the ProphitBet ML engine.

## Architecture

- **Backend**: FastAPI (Python) — REST API, JWT auth, Stripe billing
- **Frontend**: Next.js 14 (App Router) — Tailwind CSS, dark theme
- **Database**: PostgreSQL — users, subscriptions, predictions, models
- **Cache/Queue**: Redis + Celery — background tasks, rate limiting
- **Storage**: S3/MinIO — trained models, league CSVs
- **ML Core**: Reuses existing `src/` package (scikit-learn, XGBoost, Optuna, SHAP)

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 20+ (for frontend dev)
- Python 3.11+ (for backend dev)

### 1. Environment Setup

```bash
cd prophitbet-saas
cp .env.example .env
# Edit .env with your Stripe keys, secrets, etc.
```

### 2. Start with Docker Compose

```bash
docker-compose up -d
```

This starts: PostgreSQL, Redis, MinIO, FastAPI backend, Celery worker/beat, and Next.js frontend.

- **Backend API**: http://localhost:8000 (docs at /docs)
- **Frontend**: http://localhost:3000
- **MinIO Console**: http://localhost:9001

### 3. Seed League Data

```bash
# Via the API or Celery task:
curl -X POST http://localhost:8000/admin/sync-leagues \
  -H "Authorization: Bearer <admin-token>"
```

### Local Development (without Docker)

**Backend:**
```bash
cd prophitbet-saas/backend
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

**Celery Worker:**
```bash
celery -A backend.app.workers.celery_app worker --loglevel=info
```

**Frontend:**
```bash
cd prophitbet-saas/frontend
npm install
npm run dev
```

## Project Structure

```
prophitbet-saas/
  backend/
    app/
      main.py              # FastAPI entry point
      config.py            # Settings (env vars)
      middleware.py         # Rate limiting, logging
      auth/                # JWT, OAuth, dependencies
      api/                 # Route modules (auth, predictions, leagues, models, analysis, fixtures, billing, admin)
      db/                  # SQLAlchemy models + session
      services/            # Business logic (prediction, league, model, analysis, email, explainability)
      workers/             # Celery tasks (data sync, training, predictions, fixture scraping)
    alembic/               # DB migrations
    requirements.txt
    Dockerfile
  frontend/
    app/                   # Next.js App Router pages
      page.tsx             # Landing page
      pricing/             # Pricing page
      login/               # Login page
      register/            # Registration page
      dashboard/           # User dashboard
      predictions/         # Predictions list + detail
      leagues/             # League browser + detail
      analysis/            # Interactive analysis tools
      models/              # Model list, train, detail + explainability
      fixtures/            # Upcoming fixtures
      billing/             # Subscription management
      settings/            # User settings
      admin/               # Admin panel
    components/            # Shared UI components
    lib/                   # API client, auth context, utilities
    package.json
    Dockerfile
  docker-compose.yml
  .env.example
```

## Tier Structure

| Feature | Free | Pro ($9.99/mo) | Elite ($24.99/mo) |
|---------|------|----------------|---------------------|
| Daily predictions | Top 5 leagues | All 36 leagues | All 36 leagues |
| Prediction types | Result only | Result + Over/Under | Result + Over/Under |
| Probabilities | No | Yes | Yes |
| Analysis tools | Descriptive only | All 8 types | All 8 + explainability |
| Custom models | No | 3/month | Unlimited |
| Model types | — | 4 basic | All 8 |
| Auto-Tune | — | Yes | Yes |
| Fixtures | No | Yes | Yes |
| API access | No | No | Yes (1K/day) |

## API Endpoints

Full OpenAPI docs available at `http://localhost:8000/docs` when running.

Key endpoints:
- `POST /auth/register`, `POST /auth/login`, `GET /auth/me`
- `GET /predictions/today`, `GET /predictions/history`, `GET /predictions/accuracy/stats`
- `GET /leagues`, `GET /leagues/{id}/stats`, `GET /leagues/{id}/table`
- `POST /analysis` (8 analysis types)
- `POST /models/train`, `POST /models/auto-tune`, `GET /models/{id}/explain`
- `GET /fixtures/upcoming`
- `POST /billing/checkout`, `POST /billing/webhook`

## Background Tasks (Celery Beat Schedule)

| Task | Schedule | Description |
|------|----------|-------------|
| Fixture scraper | 05:00 UTC | Scrape upcoming fixtures (httpx + BeautifulSoup) |
| Data sync | 06:00 UTC | Download/update match data for all leagues |
| Prediction generation | 07:00 UTC | Generate predictions using house models |

## License

For entertainment purposes only. Not financial advice.
# Security and schema migrations

The backend applies Alembic migrations before starting under Docker Compose. For a
manual deployment, run `cd backend && alembic upgrade head` before starting the API.
Set `ENVIRONMENT=production` and provide non-default `SECRET_KEY`, `S3_ACCESS_KEY`,
and `S3_SECRET_KEY` values; production startup rejects insecure defaults.

Model artifacts use the non-executable `.skops` format. Existing `.pkl` model
records are intentionally incompatible and must be replaced by running **Train
House Models** and retraining any user models after this upgrade.

Qdrant is managed by the same Docker Compose project and persists its collections
in `C:\Users\p3rc\qdrant_storage` (the existing standalone container's data directory). The API waits briefly for it at startup, creates the
`team_profiles` collection, and idempotently seeds the built-in tactical profiles.
