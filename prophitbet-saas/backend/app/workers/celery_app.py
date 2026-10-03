import sys
from pathlib import Path

from celery import Celery
from celery.schedules import crontab

from backend.app.config import get_settings, setup_ml_path
from backend.app.workers import monitoring  # noqa: F401 - registers signal handlers

settings = get_settings()
setup_ml_path()

celery_app = Celery(
    "prophitbet",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_deduplicate_successful_tasks=True,
    # Keep network/data refresh jobs independent of CPU-heavy ML jobs.
    task_routes={"backend.app.workers.data_sync.*": {"queue": "data_sync"}},
    # Long model jobs must not be redelivered after Redis's one-hour default.
    broker_transport_options={"visibility_timeout": 86400},
    result_backend_transport_options={"visibility_timeout": 86400},
    visibility_timeout=86400,
    result_expires=86400,
)

celery_app.conf.beat_schedule = {
    "scrape-fixtures-daily": {
        "task": "backend.app.workers.fixtures.scrape_all_fixtures_task",
        "schedule": crontab(hour="*/6", minute=0),
    },
    "sync-league-data-daily": {
        "task": "backend.app.workers.data_sync.sync_all_leagues_task",
        "schedule": crontab(hour=6, minute=0),
    },
    "generate-predictions-daily": {
        "task": "backend.app.workers.predict.generate_daily_predictions_task",
        "schedule": crontab(hour=7, minute=0),
    },
    "batch-simulate-mirofish-daily": {
        "task": "backend.app.workers.mirofish_worker.batch_simulate_upcoming_task",
        "schedule": crontab(hour=7, minute=30),
        "kwargs": {"limit": 30},
    },
    "update-match-results-matchdays": {
        "task": "backend.app.workers.results.update_match_results_task",
        "schedule": crontab(minute=0, hour="13,15,17,19,21", day_of_week="tue,wed,sat,sun"),
    },
    "update-match-results-daily": {
        "task": "backend.app.workers.results.update_match_results_task",
        "schedule": crontab(hour=23, minute=0),
    },
    "train-house-models-monthly": {
        "task": "backend.app.workers.train.train_house_models_task",
        "schedule": crontab(day_of_month=1, hour=4, minute=0),
    },
    "sync-sa-odds-periodic": {
        "task": "backend.app.workers.sa_odds_worker.sync_sa_odds_task",
        "schedule": crontab(minute="*/15"),
    },
}

celery_app.autodiscover_tasks([
    "backend.app.workers.data_sync",
    "backend.app.workers.train",
    "backend.app.workers.predict",
    "backend.app.workers.fixtures",
    "backend.app.workers.results",
    "backend.app.workers.mirofish_worker",
    "backend.app.workers.sa_odds_worker",
])

from celery.signals import worker_process_init  # noqa: E402


@worker_process_init.connect
def _ensure_s3_bucket_on_worker_process(**kwargs):
    try:
        from backend.app.services.league_service import ensure_s3_bucket_sync

        ensure_s3_bucket_sync()
    except Exception:
        pass
