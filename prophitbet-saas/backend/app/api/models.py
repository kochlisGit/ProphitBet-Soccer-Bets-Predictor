from typing import Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import get_current_user, require_plan
from backend.app.db.models import TrainedModel, TrainingJob, User
from backend.app.db.session import get_db

router = APIRouter()

AVAILABLE_MODEL_TYPES = {
    "logistic": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Machine",
    "random_forest": "Random Forest",
    "xgboost": "Extreme Gradient Boosting",
    "discriminant": "Discriminant Analysis",
}

PRO_MODELS = {"logistic", "decision_tree", "naive_bayes", "knn"}
ELITE_MODELS = PRO_MODELS | {"svm", "random_forest", "xgboost", "discriminant"}

MONTHLY_MODEL_LIMITS = {"free": 0, "pro": 3, "elite": 999999}


class TrainRequest(BaseModel):
    league_id: int
    model_type: str
    target_type: Literal["result", "over-under", "over_under", "goals-1.5", "goals-3.5", "goals-4.5", "btts", "home-goals-0.5", "home-goals-1.5", "away-goals-0.5", "away-goals-1.5", "corners-8.5", "corners-9.5", "corners-10.5", "corners-11.5", "shots-target-7.5", "shots-target-8.5"] = "result"
    hyperparams: Optional[dict] = None


class AutoTuneRequest(BaseModel):
    league_id: int
    model_type: str
    target_type: Literal["result", "over-under", "over_under", "goals-1.5", "goals-3.5", "goals-4.5", "btts", "home-goals-0.5", "home-goals-1.5", "away-goals-0.5", "away-goals-1.5", "corners-8.5", "corners-9.5", "corners-10.5", "corners-11.5", "shots-target-7.5", "shots-target-8.5"] = "result"
    n_trials: int = Field(default=50, ge=1, le=100)
    fixed_params: Optional[dict] = None


class ModelOut(BaseModel):
    id: str
    league_id: int
    model_type: str
    target_type: str
    hyperparams: Optional[dict] = None
    metrics: Optional[dict] = None
    is_house_model: bool
    created_at: str


class TrainJobOut(BaseModel):
    job_id: str
    status: str
    message: str


def _user_plan(user: User) -> str:
    if user.is_admin:
        return "elite"
    if user.subscription and user.subscription.status == "active":
        return user.subscription.plan
    return "free"


async def _reserve_training(db, user):
    """Serialize quota checks per user and count queued/running jobs as slots."""
    from datetime import datetime, timezone
    from uuid import uuid4
    await db.execute(select(User.id).where(User.id == user.id).with_for_update())
    month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    pending = (await db.execute(select(func.count()).select_from(TrainingJob).where(
        TrainingJob.user_id == user.id, TrainingJob.status.in_(["queued", "running"])
    ))).scalar() or 0
    completed = (await db.execute(select(func.count()).select_from(TrainedModel).where(
        TrainedModel.user_id == user.id, TrainedModel.is_house_model.is_(False),
        TrainedModel.created_at >= month_start
    ))).scalar() or 0
    if pending >= 2:
        raise HTTPException(status_code=429, detail="Two training jobs are already queued or running")
    limit = MONTHLY_MODEL_LIMITS[_user_plan(user)]
    if not user.is_admin and completed + pending >= limit:
        raise HTTPException(status_code=429, detail=f"Monthly model training limit reached ({limit})")
    job = TrainingJob(id=uuid4(), user_id=user.id, status="queued")
    db.add(job)
    await db.commit()  # Worker must be able to see its reservation before enqueue.
    return job


async def _dispatch_training(db, user, task, **kwargs):
    reservation = await _reserve_training(db, user)
    try:
        task.apply_async(kwargs=kwargs, task_id=str(reservation.id), soft_time_limit=1800, time_limit=1860)
    except Exception:
        reservation.status = "failed"
        await db.commit()
        raise HTTPException(status_code=503, detail="Training queue is unavailable; no slot was consumed")
    return TrainJobOut(job_id=str(reservation.id), status="queued", message="Training queued")


@router.get("", response_model=list[ModelOut])
async def list_models(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    stmt = (
        select(TrainedModel)
        .where(or_(TrainedModel.user_id == user.id, TrainedModel.is_house_model == True))
        .order_by(TrainedModel.created_at.desc())
    )
    result = await db.execute(stmt)
    models = result.scalars().all()
    return [
        ModelOut(
            id=str(m.id),
            league_id=m.league_id,
            model_type=m.model_type,
            target_type=m.target_type,
            hyperparams=m.hyperparams,
            metrics=m.metrics,
            is_house_model=m.is_house_model,
            created_at=m.created_at.isoformat(),
        )
        for m in models
    ]


@router.post("/train", response_model=TrainJobOut, status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_plan("pro")),
):
    plan = _user_plan(user)
    allowed = ELITE_MODELS if plan == "elite" else PRO_MODELS
    if body.model_type not in allowed:
        raise HTTPException(status_code=403, detail=f"Model type '{body.model_type}' not available on your plan")

    from backend.app.workers.train import train_model_task
    return await _dispatch_training(db, user, train_model_task,
        user_id=str(user.id),
        league_id=body.league_id,
        model_type=body.model_type,
        target_type=body.target_type,
        hyperparams=body.hyperparams,
    )


@router.post("/auto-tune", response_model=TrainJobOut, status_code=status.HTTP_202_ACCEPTED)
async def auto_tune(
    body: AutoTuneRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_plan("pro")),
):
    plan = _user_plan(user)
    allowed = ELITE_MODELS if plan == "elite" else PRO_MODELS
    if body.model_type not in allowed:
        raise HTTPException(status_code=403, detail=f"Model type '{body.model_type}' not available on your plan")

    from backend.app.workers.train import auto_tune_task
    return await _dispatch_training(db, user, auto_tune_task,
        user_id=str(user.id),
        league_id=body.league_id,
        model_type=body.model_type,
        target_type=body.target_type,
        n_trials=body.n_trials,
        fixed_params=body.fixed_params,
    )


@router.get("/{model_id}/status")
async def model_status(model_id: str, _user: User = Depends(get_current_user)):
    from backend.app.workers.celery_app import celery_app
    result = celery_app.AsyncResult(model_id)
    meta = result.info if isinstance(result.info, dict) else {}
    current = int(meta.get("current", 0) or 0)
    total = int(meta.get("total", 0) or 0)
    return {
        "job_id": model_id,
        "status": result.status,
        "current": current,
        "total": total,
        "percent": 100 if result.successful() else (round(current * 100 / total) if total else 0),
        "phase": meta.get("phase") or ("Complete" if result.successful() else "Queued"),
        "item": meta.get("item"),
        "result": result.result if result.successful() else None,
        "error": str(result.result) if result.failed() else None,
    }


@router.get("/{model_id}", response_model=ModelOut)
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        mid = UUID(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    result = await db.execute(select(TrainedModel).where(TrainedModel.id == mid))
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if model.user_id != user.id and not model.is_public and not model.is_house_model and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")

    return ModelOut(
        id=str(model.id),
        league_id=model.league_id,
        model_type=model.model_type,
        target_type=model.target_type,
        hyperparams=model.hyperparams,
        metrics=model.metrics,
        is_house_model=model.is_house_model,
        created_at=model.created_at.isoformat(),
    )


@router.get("/{model_id}/explain")
async def explain_model(
    model_id: str,
    plot_type: str = "bar",
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_plan("elite")),
):
    """Generate a SHAP explainability plot for a trained model (Elite only)."""
    import base64
    try:
        mid = UUID(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    result = await db.execute(select(TrainedModel).where(TrainedModel.id == mid))
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if model.user_id != user.id and not model.is_public and not model.is_house_model and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")

    from backend.app.services.league_service import get_league_dataframe, download_model_from_s3
    df = await get_league_dataframe(model.league_id, db)
    if df is None:
        raise HTTPException(status_code=404, detail="No league data available")

    model_bytes = await download_model_from_s3(model.file_path)

    from backend.app.services.explainability_service import generate_shap_plot
    plot_bytes = generate_shap_plot(model_bytes=model_bytes, df=df, plot_type=plot_type)
    img_b64 = base64.b64encode(plot_bytes).decode("utf-8")

    return {
        "model_id": model_id,
        "plot_type": plot_type,
        "image_base64": img_b64,
    }


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        mid = UUID(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    result = await db.execute(select(TrainedModel).where(TrainedModel.id == mid))
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if model.user_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")

    if model.file_path:
        try:
            from backend.app.services.league_service import delete_model_from_s3
            await delete_model_from_s3(model.file_path)
        except Exception:
            pass

    await db.delete(model)
