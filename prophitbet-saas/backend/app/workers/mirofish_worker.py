import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _create_isolated_async_session():
    """Create a process-isolated async session with NullPool to prevent prefork connection sharing."""
    from backend.app.config import get_settings
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return session_factory(), engine


@celery_app.task(name="backend.app.workers.mirofish_worker.simulate_prediction_task", bind=True)
def simulate_prediction_task(self, prediction_id: str, force_recompute: bool = False) -> Dict[str, Any]:
    """Run MiroFish multi-agent simulation asynchronously for a single prediction."""
    from backend.app.services.mirofish_service import MiroFishService

    async def _run():
        session, engine = _create_isolated_async_session()
        try:
            async with session as db:
                pred_uuid = uuid.UUID(prediction_id) if isinstance(prediction_id, str) else prediction_id
                sim = await MiroFishService.simulate_prediction(
                    prediction_id=pred_uuid,
                    db=db,
                    force_recompute=force_recompute,
                )
                return {
                    "prediction_id": str(sim.prediction_id),
                    "swarm_predicted_result": sim.swarm_predicted_result,
                    "ensemble_predicted_result": sim.ensemble_predicted_result,
                    "consensus_level": sim.consensus_level,
                    "confidence_score": sim.confidence_score,
                }
        finally:
            await engine.dispose()

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error(f"Celery MiroFish simulation failed for {prediction_id}: {exc}", exc_info=True)
        raise


@celery_app.task(name="backend.app.workers.mirofish_worker.batch_simulate_upcoming_task", bind=True)
def batch_simulate_upcoming_task(self, limit: int = 20) -> Dict[str, Any]:
    """Batch-simulate upcoming predictions that do not yet have a MiroFish simulation."""
    from sqlalchemy import select
    from backend.app.db.models import MiroFishSimulation, Prediction
    from backend.app.services.mirofish_service import MiroFishService

    async def _run():
        now = datetime.now(timezone.utc)
        upcoming_threshold = now - timedelta(hours=36)
        simulated_count = 0
        results: List[Dict[str, Any]] = []

        session, engine = _create_isolated_async_session()
        try:
            async with session as db:
                stmt = (
                    select(Prediction).where(Prediction.market_type == "result")
                    .outerjoin(MiroFishSimulation, Prediction.id == MiroFishSimulation.prediction_id)
                    .where(
                        Prediction.match_date >= upcoming_threshold,
                        MiroFishSimulation.id.is_(None),
                    )
                    .order_by(Prediction.match_date.asc())
                    .limit(limit)
                )
                preds = (await db.execute(stmt)).scalars().all()
                logger.info(f"Celery Batch MiroFish: processing {len(preds)} pending upcoming matches")

                self.update_state(state="PROGRESS", meta={
                    "current": 0, "total": len(preds),
                    "phase": "Preparing swarm simulations", "item": None,
                })

                for index, pred in enumerate(preds, start=1):
                    item = f"{pred.home_team} vs {pred.away_team}"
                    self.update_state(state="PROGRESS", meta={
                        "current": index - 1, "total": len(preds),
                        "phase": "Running swarm simulation", "item": item,
                    })
                    try:
                        sim = await MiroFishService.simulate_prediction(
                            prediction_id=pred.id,
                            db=db,
                            force_recompute=False,
                        )
                        simulated_count += 1
                        results.append({
                            "prediction_id": str(pred.id),
                            "match": f"{pred.home_team} vs {pred.away_team}",
                            "ensemble": sim.ensemble_predicted_result,
                            "consensus": sim.consensus_level,
                        })
                    except Exception as exc:
                        logger.error(f"Error in batch simulation for {pred.id}: {exc}")
                    finally:
                        self.update_state(state="PROGRESS", meta={
                            "current": index, "total": len(preds),
                            "phase": "Simulation complete", "item": item,
                        })

            return {
                "total_processed": len(preds),
                "total_simulated": simulated_count,
                "results": results,
            }
        finally:
            await engine.dispose()

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error(f"Celery Batch MiroFish task failed: {exc}", exc_info=True)
        raise
