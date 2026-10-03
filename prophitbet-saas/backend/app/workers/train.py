import asyncio
import logging
from uuid import UUID

from backend.app.config import setup_ml_path
from backend.app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _set_training_status(session, task_id, status, commit=False):
    from backend.app.db.models import TrainingJob
    job = session.get(TrainingJob, UUID(task_id))
    if job:
        job.status = status
    if commit:
        session.commit()


def _get_sync_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.config import get_settings
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL_SYNC)
    return sessionmaker(bind=engine)()


def _load_league_df(league_id: int, session):
    """Load the league's latest dataset DataFrame."""
    from backend.app.db.models import LeagueDataset
    ds = (
        session.query(LeagueDataset)
        .filter(LeagueDataset.league_id == league_id)
        .order_by(LeagueDataset.created_at.desc())
        .first()
    )
    if ds is None:
        raise ValueError(f"No dataset found for league {league_id}")

    loop = asyncio.new_event_loop()
    try:
        from backend.app.services.league_service import _load_csv_from_s3
        df = loop.run_until_complete(_load_csv_from_s3(ds.file_path))
    finally:
        loop.close()
    return df


@celery_app.task(name="backend.app.workers.train.train_model_task", bind=True)
def train_model_task(self, user_id: str, league_id: int, model_type: str,
                     target_type: str = "result", hyperparams: dict = None):
    """Train a single model and store it in S3 + DB."""
    setup_ml_path()
    session = _get_sync_session()
    try:
        _set_training_status(session, self.request.id, "running", commit=True)
        self.update_state(state="PROGRESS", meta={"current": 0, "total": 4, "phase": "Loading league data", "item": None})
        df = _load_league_df(league_id, session)
        logger.info(f"Training {model_type} for league {league_id}, rows={len(df)}")

        from backend.app.services.model_service import train_model
        self.update_state(state="PROGRESS", meta={"current": 1, "total": 4, "phase": "Training and validating model", "item": model_type})
        result = train_model(
            df=df,
            model_type=model_type,
            target_type=target_type,
            hyperparams=hyperparams,
        )

        model_bytes = result["model_bytes"]
        s3_key = f"models/{user_id}/{league_id}/{model_type}/{self.request.id}.skops"
        self.update_state(state="PROGRESS", meta={"current": 2, "total": 4, "phase": "Uploading safe model artifact", "item": model_type})

        loop = asyncio.new_event_loop()
        try:
            from backend.app.services.league_service import upload_model_to_s3
            loop.run_until_complete(upload_model_to_s3(model_bytes, s3_key))
        finally:
            loop.close()

        from backend.app.db.models import TrainedModel
        model_record = TrainedModel(
            user_id=UUID(user_id),
            league_id=league_id,
            model_type=model_type,
            target_type=target_type,
            hyperparams=result["hyperparams"],
            metrics=result["metrics"],
            file_path=s3_key,
            is_public=False,
            is_house_model=False,
        )
        session.add(model_record)
        self.update_state(state="PROGRESS", meta={"current": 3, "total": 4, "phase": "Saving model record", "item": model_type})
        _set_training_status(session, self.request.id, "succeeded")
        session.commit()

        logger.info(f"Model trained and saved: {model_record.id}")
        return {
            "model_id": str(model_record.id),
            "metrics": result["metrics"],
            "status": "completed",
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        session.rollback()
        _set_training_status(session, self.request.id, "failed", commit=True)
        raise
    finally:
        session.close()


@celery_app.task(name="backend.app.workers.train.train_house_models_task", bind=True)
def train_house_models_task(self):
    """Train independent house models only where observed target data exists."""
    setup_ml_path()
    from src.preprocessing.utils.target import TargetType, construct_targets
    from backend.app.db.models import League, TrainedModel
    from backend.app.services.model_service import train_model, prepare_target_data
    from backend.app.services.league_service import upload_model_to_s3
    session = _get_sync_session()
    trained, skipped = 0, []
    try:
        leagues = session.query(League).filter(League.is_active.is_(True)).all()
        targets = list(TargetType)
        total = len(leagues) * len(targets)
        current = 0
        for league in leagues:
            try:
                df = _load_league_df(league.id, session)
            except Exception as exc:
                skipped.append(dict(league_id=league.id, market="all", reason=str(exc)))
                session.rollback()
                current += len(targets)
                continue
            for target in targets:
                self.update_state(state="PROGRESS", meta=dict(current=current, total=total,
                    phase="Training market model", item=f"{league.name} · {target.value}"))
                try:
                    eligible = prepare_target_data(df, target).drop(columns=['Source', 'ScoreScope', '1', 'X', '2'], errors='ignore').dropna()
                    if len(eligible) < 50 or len(set(construct_targets(eligible, target))) < 2:
                        raise ValueError("Need at least 50 complete observed rows and two outcome classes")
                    result = train_model(df=df, model_type="random_forest", target_type=target.value, use_odds=False)
                    holdout = result['metrics']['temporal_report']['folds'][-1]['model']
                    if holdout['samples'] < 50 or not result['metrics']['beats_frequency_baseline']:
                        raise ValueError('Candidate not promoted: need 50 holdout matches and lower log loss than the frequency baseline')
                    key = f"models/house/{league.id}/{target.value}/{self.request.id}.skops"
                    asyncio.run(upload_model_to_s3(result["model_bytes"], key))
                    existing = session.query(TrainedModel).filter(
                        TrainedModel.league_id == league.id, TrainedModel.is_house_model.is_(True),
                        TrainedModel.target_type == target.value).order_by(TrainedModel.created_at.desc()).first()
                    if existing:
                        existing.metrics, existing.hyperparams, existing.file_path = result["metrics"], result["hyperparams"], key
                    else:
                        session.add(TrainedModel(user_id=None, league_id=league.id, model_type="random_forest",
                            target_type=target.value, metrics=result["metrics"], hyperparams=result["hyperparams"],
                            file_path=key, is_public=True, is_house_model=True))
                    session.commit()
                    trained += 1
                except Exception as exc:
                    session.rollback()
                    skipped.append(dict(league_id=league.id, market=target.value, reason=str(exc)))
                    logger.warning("Skipped %s %s: %s", league.name, target.value, exc)
                finally:
                    current += 1
                    self.update_state(state="PROGRESS", meta=dict(current=current, total=total,
                        phase="Market checked", item=f"{league.name} · {target.value}"))
        return dict(trained=trained, total=total, skipped=skipped)
    finally:
        session.close()


@celery_app.task(name="backend.app.workers.train.auto_tune_task", bind=True)
def auto_tune_task(self, user_id: str, league_id: int, model_type: str,
                   target_type: str = "result", n_trials: int = 50,
                   fixed_params: dict = None):
    """Run Optuna auto-tune then train the best model."""
    setup_ml_path()
    session = _get_sync_session()
    try:
        _set_training_status(session, self.request.id, "running", commit=True)
        total_steps = n_trials + 3
        self.update_state(state="PROGRESS", meta={"current": 0, "total": total_steps, "phase": "Loading league data", "item": None})
        df = _load_league_df(league_id, session)
        logger.info(f"Auto-tuning {model_type} for league {league_id}, trials={n_trials}")

        from backend.app.services.model_service import auto_tune_model
        result = auto_tune_model(
            df=df,
            model_type=model_type,
            target_type=target_type,
            n_trials=n_trials,
            fixed_params=fixed_params,
            progress_callback=lambda completed, total: self.update_state(
                state="PROGRESS",
                meta={
                    "current": completed,
                    "total": total_steps,
                    "phase": "Testing hyperparameter trials",
                    "item": f"Trial {completed} of {total}",
                },
            ),
        )

        model_bytes = result["model_bytes"]
        s3_key = f"models/{user_id}/{league_id}/{model_type}/tuned-{self.request.id}.skops"
        self.update_state(state="PROGRESS", meta={"current": n_trials + 1, "total": total_steps, "phase": "Uploading tuned model", "item": model_type})

        loop = asyncio.new_event_loop()
        try:
            from backend.app.services.league_service import upload_model_to_s3
            loop.run_until_complete(upload_model_to_s3(model_bytes, s3_key))
        finally:
            loop.close()

        from backend.app.db.models import TrainedModel
        model_record = TrainedModel(
            user_id=UUID(user_id),
            league_id=league_id,
            model_type=model_type,
            target_type=target_type,
            hyperparams=result["hyperparams"],
            metrics=result["metrics"],
            file_path=s3_key,
            is_public=False,
            is_house_model=False,
        )
        session.add(model_record)
        self.update_state(state="PROGRESS", meta={"current": n_trials + 2, "total": total_steps, "phase": "Saving tuned model", "item": model_type})
        _set_training_status(session, self.request.id, "succeeded")
        session.commit()

        logger.info(f"Auto-tuned model saved: {model_record.id}")
        return {
            "model_id": str(model_record.id),
            "metrics": result["metrics"],
            "best_trial": result.get("best_trial"),
            "status": "completed",
        }
    except Exception as e:
        logger.error(f"Auto-tune failed: {e}")
        session.rollback()
        _set_training_status(session, self.request.id, "failed", commit=True)
        raise
    finally:
        session.close()
