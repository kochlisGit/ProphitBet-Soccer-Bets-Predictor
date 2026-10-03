import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import or_
from concurrent.futures import ThreadPoolExecutor

from backend.app.config import setup_ml_path
from backend.app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Thread pool for blocking operations
_thread_pool: Optional[ThreadPoolExecutor] = None


def _get_thread_pool() -> ThreadPoolExecutor:
    """Get or create thread pool for blocking operations."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=4)
    return _thread_pool


def _get_sync_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.config import get_settings
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL_SYNC)
    return sessionmaker(bind=engine)()


@celery_app.task(name="backend.app.workers.predict.generate_daily_predictions_task", bind=True)
def generate_daily_predictions_task(self, publish_external: bool = False):
    from celery.exceptions import Ignore
    from backend.app.workers.task_guard import prediction_run_guard
    with prediction_run_guard(self.request.id) as (acquired, owner):
        if not acquired:
            logger.info("Prediction run already active: %s; skipping %s", owner, self.request.id)
            if owner == str(self.request.id):
                # A redelivery of the same ID must not overwrite its active result.
                raise Ignore()
            return {"status": "already_running", "active_job_id": owner,
                    "message": "Another prediction run is active; no overlapping work started."}
        return _generate_daily_predictions(self, publish_external)


def _generate_daily_predictions(self, publish_external: bool = False):
    """Generate predictions for all active leagues with upcoming fixtures using house models."""
    setup_ml_path()
    session = _get_sync_session()
    try:
        from backend.app.db.models import (
            Fixture, League, LeagueDataset, Prediction, TrainedModel,
        )
        from backend.app.services.prediction_service import generate_predictions_for_league
        from backend.app.services.league_service import download_model_from_s3_sync, _load_csv_from_s3_sync

        leagues = session.query(League).filter(League.is_active == True).all()
        now = datetime.now(timezone.utc)
        total_predictions = 0
        failures = []

        self.update_state(state="PROGRESS", meta={
            "current": 0, "total": len(leagues), "phase": "Preparing prediction run", "item": None,
        })

        for index, lg in enumerate(leagues, start=1):
            item = f"{lg.country} — {lg.name}"
            self.update_state(state="PROGRESS", meta={
                "current": index - 1, "total": len(leagues), "phase": "Generating predictions", "item": item,
            })
            try:
                from sqlalchemy import text
                house_models = (
                    session.query(TrainedModel)
                    .filter(
                        TrainedModel.league_id == lg.id,
                        TrainedModel.is_house_model == True,
                    )
                    .order_by(TrainedModel.created_at.desc())
                    .all()
                )
                if not house_models:
                    logger.debug(f"No house model for {lg.country} - {lg.name}, skipping")
                    continue

                fixtures = (
                    session.query(Fixture)
                    .filter(
                        Fixture.league_id == lg.id,
                        Fixture.is_current.is_(True),
                        Fixture.match_date >= now,
                        Fixture.match_date <= now + timedelta(days=7),
                        Fixture.source_url.isnot(None),
                        Fixture.fetched_at >= now - timedelta(hours=48),
                    )
                    .all()
                )
                if not fixtures:
                    continue

                ds = (
                    session.query(LeagueDataset)
                    .filter(LeagueDataset.league_id == lg.id)
                    .order_by(LeagueDataset.created_at.desc())
                    .first()
                )
                if ds is None:
                    continue

                league_df = _load_csv_from_s3_sync(ds.file_path)
                seen_markets = set()
                for house_model in house_models:
                    if house_model.target_type in seen_markets:
                        continue
                    seen_markets.add(house_model.target_type)
                    try:
                        session.execute(text("SELECT pg_advisory_xact_lock(718310)"))
                        model_bytes = download_model_from_s3_sync(house_model.file_path)
        
        
                        fixture_dicts = [
                            {
                                "fixture_id": str(f.id),
                                "home_team": f.home_team,
                                "away_team": f.away_team,
                                "odds_1": f.odds_1,
                                "odds_x": f.odds_x,
                                "odds_2": f.odds_2,
                            }
                            for f in fixtures
                        ]
        
                        predictions = generate_predictions_for_league(
                            model_bytes=model_bytes,
                            league_df=league_df,
                            fixtures=fixture_dicts,
                            target_type=house_model.target_type,
                        )
        
                        fixture_map = {str(f.id): f for f in fixtures}
                        for pred in predictions:
                            fixture = fixture_map.get(pred["fixture_id"])
                            if fixture is None:
                                continue
        
                            existing = session.query(Prediction).filter(
                                Prediction.model_id == house_model.id, Prediction.league_id == lg.id,
                                Prediction.home_team == fixture.home_team, Prediction.away_team == fixture.away_team,
                                Prediction.match_date == fixture.match_date,
                                Prediction.actual_result.is_(None),
                            ).first()
                            if existing:
                                existing.predicted_result = pred["predicted_result"]
                                existing.probabilities = pred["probabilities"]
                                existing.created_at = datetime.now(timezone.utc)
                                fixture.predicted = True
                                total_predictions += 1
                                continue
        
                            db_pred = Prediction(
                                model_id=house_model.id,
                                league_id=lg.id,
                                home_team=pred["home_team"],
                                away_team=pred["away_team"],
                                match_date=fixture.match_date,
                                predicted_result=pred["predicted_result"],
                                probabilities=pred["probabilities"],
                            )
                            session.add(db_pred)
        
                            if fixture:
                                fixture.predicted = True
        
                            total_predictions += 1
        
                        session.commit()
                        logger.info(f"Generated {len(predictions)} predictions for {lg.country} - {lg.name}")
        
                        try:
                            from backend.app.metrics import PREDICTIONS_GENERATED
                            PREDICTIONS_GENERATED.labels(
                                league=lg.name or "unknown",
                                market=house_model.target_type or "full_time"
                            ).inc(len(predictions))
                        except Exception as met_err:
                            logger.debug("Could not record prometheus prediction metrics: %s", met_err)
        
                        # External publication requires an explicit opt-in; normal
                        # admin/scheduled prediction generation stays inside the app.
                        if not publish_external:
                            continue
                        try:
                            from backend.app.services.linear_service import LinearService
                            for p_info in predictions:
                                probs = p_info.get("probabilities", {})
                                pick = p_info.get("predicted_result", "H")
                                pick_prob = float(probs.get(pick, 0.0))
                                observed = fixture_map.get(p_info.get("fixture_id"))
                                odds_field = {"H": "odds_1", "D": "odds_x", "A": "odds_2"}.get(pick)
                                odds = getattr(observed, odds_field, None) if odds_field and observed else None
                                if odds is None or odds <= 1:
                                    continue
                                if pick_prob >= 0.50:
                                    alert_data = {
                                        "match": f"{p_info['home_team']} vs {p_info['away_team']}",
                                        "league": f"{lg.country} - {lg.name}",
                                        "match_date": observed.match_date.isoformat(),
                                        "pick": pick,
                                        "pick_name": "Home Win" if pick == "H" else ("Draw" if pick == "D" else "Away Win"),
                                        "market_odds": odds,
                                        "win_probability_pct": round(pick_prob * 100.0, 1),
                                        "ev_edge_pct": round((pick_prob * odds - 1.0) * 100.0, 1),
                                        "quarter_kelly_stake_pct": round(max(0.0, (pick_prob * odds - 1.0) / (odds - 1.0)) * 25, 2),
                                        "consensus_level": "MODEL_PREDICTION"
                                    }
                                    LinearService.create_prediction_issue(alert_data)
                        except Exception as e:
                            logger.warning("Could not dispatch predictions to Linear: %s", e)
        
                    except Exception as exc:
                        session.rollback()
                        failures.append(dict(league_id=lg.id, market=house_model.target_type, error=str(exc)))

            except Exception as e:
                logger.error(f"Prediction generation failed for {lg.country} - {lg.name}: {e}")
                failures.append({"league_id": lg.id, "error": str(e)})
                session.rollback()
                continue
            finally:
                self.update_state(state="PROGRESS", meta={
                    "current": index, "total": len(leagues), "phase": "League checked", "item": item,
                })

        if failures and total_predictions == 0:
            raise RuntimeError(f"Prediction generation failed for {len(failures)} leagues: {failures[0]['error']}")
        return {"total_predictions": total_predictions, "failures": failures,
                "coverage": "partial" if failures else "complete"}
    finally:
        session.close()
        # Invalidate predictions cache after generation
        try:
            from backend.app.services.cache import invalidate_predictions_cache
            invalidate_predictions_cache()
        except Exception as e:
            logger.warning(f"Failed to invalidate predictions cache: {e}")
