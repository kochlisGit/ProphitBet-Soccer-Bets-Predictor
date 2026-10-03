"""Celery Worker Task for South African Bookmaker Odds Synchronization."""

import asyncio
import logging
from backend.app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="backend.app.workers.sa_odds_worker.sync_sa_odds_task")
def sync_sa_odds_task():
    """Periodic Celery worker task to scrape and synchronize Hollywoodbets & Betway SA odds."""
    logger.info("Executing periodic SA odds synchronization task (Hollywoodbets & Betway)...")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from backend.app.config import get_settings
    from backend.app.services.sa_odds_service import sa_odds_service

    async def _run():
        # Celery forks must not reuse the API process's async engine/loop.
        engine = create_async_engine(get_settings().DATABASE_URL)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with factory() as session:
                return await sa_odds_service.sync_all_sa_markets(session)
        finally:
            await engine.dispose()

    try:
        res = asyncio.run(_run())
        logger.info(f"SA odds task finished successfully: {res}")
        return res
    except Exception as e:
        logger.error(f"Error in sync_sa_odds_task: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
