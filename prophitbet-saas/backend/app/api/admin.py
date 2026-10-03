from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import require_admin
from backend.app.db.models import League, Prediction, Subscription, TrainedModel, User
from backend.app.db.session import get_db
from backend.app.workers.monitoring import get_task_metrics, get_recent_failures, check_health, check_and_alert

router = APIRouter()


@router.get("/jobs/{job_id}")
async def admin_job_status(
    job_id: str,
    _admin: User = Depends(require_admin),
):
    """Return live Celery progress metadata for an admin pipeline job."""
    from backend.app.workers.celery_app import celery_app

    task = celery_app.AsyncResult(job_id)
    meta = task.info if isinstance(task.info, dict) else {}
    current = int(meta.get("current", 0) or 0)
    total = int(meta.get("total", 0) or 0)
    percent = int(meta.get("percent", round(current * 100 / total) if total else 0))
    if task.successful():
        percent = 100
    else:
        # All work units may be processed while the DB commit is still pending.
        percent = min(percent, 99)
    return {
        "job_id": job_id,
        "status": task.status,
        "current": current,
        "total": total,
        "percent": max(0, min(100, percent)),
        "phase": meta.get("phase") or ("Complete" if task.successful() else "Queued"),
        "item": meta.get("item"),
        "result": task.result if task.successful() else None,
        "error": str(task.result) if task.failed() else None,
    }


class AdminStatsOut(BaseModel):
    total_users: int
    pro_users: int
    elite_users: int
    total_models: int
    total_predictions: int
    total_leagues: int


class AdminUserOut(BaseModel):
    id: str
    email: str
    name: str
    plan: str
    is_active: bool
    is_admin: bool
    created_at: str


@router.get("/stats", response_model=AdminStatsOut)
async def admin_stats(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    total_users = (await db.execute(select(func.count()).select_from(User))).scalar() or 0
    pro_users = (await db.execute(
        select(func.count()).select_from(Subscription).where(Subscription.plan == "pro", Subscription.status == "active")
    )).scalar() or 0
    elite_users = (await db.execute(
        select(func.count()).select_from(Subscription).where(Subscription.plan == "elite", Subscription.status == "active")
    )).scalar() or 0
    total_models = (await db.execute(select(func.count()).select_from(TrainedModel))).scalar() or 0
    total_predictions = (await db.execute(select(func.count()).select_from(Prediction))).scalar() or 0
    total_leagues = (await db.execute(select(func.count()).select_from(League).where(League.is_active == True))).scalar() or 0

    return AdminStatsOut(
        total_users=total_users,
        pro_users=pro_users,
        elite_users=elite_users,
        total_models=total_models,
        total_predictions=total_predictions,
        total_leagues=total_leagues,
    )


@router.get("/users", response_model=list[AdminUserOut])
async def admin_list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=10, le=200),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    offset = (page - 1) * per_page
    result = await db.execute(
        select(User).order_by(User.created_at.desc()).offset(offset).limit(per_page)
    )
    users = result.scalars().all()
    out = []
    for u in users:
        plan = "free"
        if u.subscription and u.subscription.status == "active":
            plan = u.subscription.plan
        out.append(AdminUserOut(
            id=str(u.id),
            email=u.email,
            name=u.name,
            plan=plan,
            is_active=u.is_active,
            is_admin=u.is_admin,
            created_at=u.created_at.isoformat(),
        ))
    return out


@router.post("/sync-leagues")
async def trigger_league_sync(
    force: bool = Query(False),
    skip_if_synced_within_hours: Optional[int] = Query(None, ge=0, le=8760),
    _admin: User = Depends(require_admin),
):
    from backend.app.workers.data_sync import sync_all_leagues_task
    kwargs = {"force_resync": force}
    if skip_if_synced_within_hours is not None:
        kwargs["skip_if_synced_within_hours"] = skip_if_synced_within_hours
    job = sync_all_leagues_task.delay(**kwargs)
    return {"job_id": job.id, "message": "League data sync triggered", "kwargs": kwargs}


@router.post("/generate-predictions")
async def trigger_predictions(
    _admin: User = Depends(require_admin),
):
    from backend.app.workers.predict import generate_daily_predictions_task
    job = generate_daily_predictions_task.delay()
    return {"job_id": job.id, "message": "Prediction generation triggered"}


@router.post("/train-house-models")
async def trigger_house_model_training(
    _admin: User = Depends(require_admin),
):
    from backend.app.workers.train import train_house_models_task
    job = train_house_models_task.delay()
    return {"job_id": job.id, "message": "House model training triggered"}


@router.post("/scrape-fixtures")
async def trigger_fixture_scraping(
    _admin: User = Depends(require_admin),
):
    from backend.app.workers.fixtures import scrape_all_fixtures_task
    job = scrape_all_fixtures_task.delay()
    return {"job_id": job.id, "message": "Fixture scraping triggered"}


@router.post("/update-results")
async def trigger_result_update(
    _admin: User = Depends(require_admin),
):
    from backend.app.workers.results import update_match_results_task
    job = update_match_results_task.delay()
    return {"job_id": job.id, "message": "Match result update triggered"}


@router.post("/mirofish/batch-simulate")
async def trigger_mirofish_batch_simulate(
    _admin: User = Depends(require_admin),
    limit: int = Query(25, ge=1, le=100),
):
    """Trigger background MiroFish multi-agent simulation for upcoming matches."""
    from backend.app.workers.mirofish_worker import batch_simulate_upcoming_task
    job = batch_simulate_upcoming_task.delay(limit=limit)
    return {
        "job_id": job.id,
        "message": f"MiroFish batch simulation triggered for upcoming matches (limit={limit})",
    }



# Monitoring endpoints
class TaskMetricsOut(BaseModel):
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    retried_tasks: int
    total_duration_seconds: float
    success_rate: float
    failure_rate: float
    avg_duration_seconds: float
    avg_duration_per_task: dict[str, float]
    recent_failures: list[dict]


@router.get("/monitoring/metrics", response_model=TaskMetricsOut)
async def get_monitoring_metrics(
    _admin: User = Depends(require_admin),
):
    """Get Celery task execution metrics."""
    metrics = get_task_metrics()
    recent_failures = get_recent_failures(20)
    return TaskMetricsOut(
        total_tasks=metrics["total_tasks"],
        successful_tasks=metrics["successful_tasks"],
        failed_tasks=metrics["failed_tasks"],
        retried_tasks=metrics["retried_tasks"],
        total_duration_seconds=metrics["total_duration_seconds"],
        success_rate=metrics["success_rate"],
        failure_rate=metrics["failure_rate"],
        avg_duration_seconds=metrics["avg_duration_seconds"],
        avg_duration_per_task=metrics["avg_duration_per_task"],
        recent_failures=recent_failures,
    )


@router.get("/monitoring/health")
async def get_monitoring_health(
    _admin: User = Depends(require_admin),
):
    """Health check for task processing system."""
    return check_health()


@router.get("/monitoring/alerts")
async def get_monitoring_alerts(
    _admin: User = Depends(require_admin),
):
    """Get active alerts for task processing."""
    alerts = check_and_alert()
    return {"alerts": alerts, "count": len(alerts)}


@router.get("/telemetry")
async def get_system_telemetry(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Comprehensive institutional system telemetry."""
    import time
    import json
    import urllib.request
    from backend.app.config import get_settings
    from backend.app.workers.celery_app import celery_app
    from backend.app.db.models import MiroFishSimulation

    settings = get_settings()

    # 1. Database stats
    total_preds = (await db.execute(select(func.count()).select_from(Prediction))).scalar() or 0
    settled_preds = (await db.execute(
        select(func.count()).select_from(Prediction).where(Prediction.actual_result.isnot(None))
    )).scalar() or 0
    correct_preds = (await db.execute(
        select(func.count()).select_from(Prediction).where(Prediction.is_correct == True)
    )).scalar() or 0
    total_sims = (await db.execute(select(func.count()).select_from(MiroFishSimulation))).scalar() or 0

    hit_rate_pct = round((correct_preds / settled_preds) * 100, 2) if settled_preds > 0 else 0.0

    # 2. Qdrant status
    qdrant_status = "error"
    qdrant_collections = []
    try:
        req = urllib.request.Request(f"{settings.QDRANT_URL}/collections", headers={"User-Agent": "ProphitBet"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                qdata = json.loads(resp.read().decode())
                qdrant_status = "connected"
                qdrant_collections = [c["name"] for c in qdata.get("result", {}).get("collections", [])]
    except Exception as e:
        qdrant_status = f"unreachable: {str(e)}"

    # 3. Redis status & ping
    redis_status = "connected"
    redis_ping_ms = 0.0
    try:
        import redis
        r = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=2)
        t0 = time.perf_counter()
        r.ping()
        redis_ping_ms = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        redis_status = f"error: {str(e)}"

    # 4. Celery beat schedule summary
    beat_schedule_keys = list(getattr(celery_app.conf, "beat_schedule", {}).keys())

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "database": {
            "total_predictions": total_preds,
            "settled_predictions": settled_preds,
            "correct_predictions": correct_preds,
            "empirical_accuracy_pct": hit_rate_pct,
            "total_mirofish_simulations": total_sims,
        },
        "qdrant_vector_db": {
            "status": qdrant_status,
            "url": settings.QDRANT_URL,
            "collections": qdrant_collections,
            "count": len(qdrant_collections),
        },
        "redis": {
            "status": redis_status,
            "ping_latency_ms": redis_ping_ms,
        },
        "celery": {
            "beat_scheduled_tasks": beat_schedule_keys,
            "beat_tasks_count": len(beat_schedule_keys),
        },
    }


from backend.app.services.alert_service import AlertService


class AlertDispatchRequest(BaseModel):
    webhook_url: str
    min_ev_pct: float = 5.0
    platform: str = "discord"


@router.get("/alerts/preview")
async def preview_value_alerts(
    min_ev: float = 4.0,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Preview formatted +EV value betting alerts across upcoming fixtures."""
    alerts = await AlertService.get_high_value_alerts(db, min_ev_pct=min_ev, limit=10)
    formatted_alerts = []
    for a in alerts:
        formatted_alerts.append({
            "alert": a,
            "discord_embed": AlertService.format_discord_webhook_payload(a),
            "telegram_markdown": AlertService.format_telegram_message(a),
        })
    return {"alerts": formatted_alerts, "count": len(formatted_alerts)}


@router.post("/alerts/dispatch")
async def dispatch_value_alert(
    payload: AlertDispatchRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Dispatch high-conviction +EV alert to a live Discord/Telegram webhook."""
    alerts = await AlertService.get_high_value_alerts(db, min_ev_pct=payload.min_ev_pct, limit=1)
    if not alerts:
        return {"status": "no_alerts", "message": f"No fixtures found matching +EV >= {payload.min_ev_pct}%"}

    success = await AlertService.dispatch_webhook(
        webhook_url=payload.webhook_url,
        alert=alerts[0],
        platform=payload.platform,
    )
    return {"status": "dispatched" if success else "failed", "alert": alerts[0]}

