"""
Celery task monitoring and alerting.

Provides hooks for tracking task execution, failures, retries, and performance metrics.
Integrates with Celery signals to automatically capture task lifecycle events.
Uses Redis for shared metrics storage across processes.
"""

import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional

from celery import signals
from celery.app.task import Task
import redis

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

# Redis keys for metrics storage
METRICS_KEY = "prophitbet:monitoring:metrics"
FAILURES_KEY = "prophitbet:monitoring:failures"
DURATIONS_KEY = "prophitbet:monitoring:durations"

_MAX_FAILURE_HISTORY = 100
_MAX_DURATION_HISTORY = 1000


def _get_redis() -> redis.Redis:
    """Get Redis connection."""
    settings = get_settings()
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


def _record_metric(task_name: str, duration: float, success: bool, error: Optional[str] = None) -> None:
    """Record task execution metrics in Redis."""
    r = _get_redis()
    
    # Use a pipeline for atomic operations
    pipe = r.pipeline()
    
    # Increment counters
    pipe.hincrby(METRICS_KEY, "total_tasks", 1)
    pipe.hincrbyfloat(METRICS_KEY, "total_duration_seconds", duration)
    if success:
        pipe.hincrby(METRICS_KEY, "successful_tasks", 1)
    else:
        pipe.hincrby(METRICS_KEY, "failed_tasks", 1)
        # Add failure record
        failure_record = {
            "task_name": task_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "duration": duration,
        }
        pipe.lpush(FAILURES_KEY, json.dumps(failure_record))
        pipe.ltrim(FAILURES_KEY, 0, _MAX_FAILURE_HISTORY - 1)
    
    # Track durations per task
    pipe.lpush(f"{DURATIONS_KEY}:{task_name}", duration)
    pipe.ltrim(f"{DURATIONS_KEY}:{task_name}", 0, _MAX_DURATION_HISTORY - 1)
    
    pipe.execute()


def get_task_metrics() -> dict:
    """Get current task metrics summary from Redis."""
    r = _get_redis()
    
    # Get basic counters
    metrics_data = r.hgetall(METRICS_KEY)
    total_tasks = int(metrics_data.get("total_tasks", 0))
    successful_tasks = int(metrics_data.get("successful_tasks", 0))
    failed_tasks = int(metrics_data.get("failed_tasks", 0))
    retried_tasks = int(metrics_data.get("retried_tasks", 0))
    total_duration = float(metrics_data.get("total_duration_seconds", 0.0))
    
    metrics = {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "retried_tasks": retried_tasks,
        "total_duration_seconds": total_duration,
        "task_failures": [],
        "task_durations": {},
    }
    
    # Calculate averages
    if total_tasks > 0:
        metrics["success_rate"] = successful_tasks / total_tasks
        metrics["failure_rate"] = failed_tasks / total_tasks
        metrics["avg_duration_seconds"] = total_duration / total_tasks
    else:
        metrics["success_rate"] = 0.0
        metrics["failure_rate"] = 0.0
        metrics["avg_duration_seconds"] = 0.0
    
    # Get recent failures
    failures_json = r.lrange(FAILURES_KEY, 0, _MAX_FAILURE_HISTORY - 1)
    metrics["task_failures"] = [json.loads(f) for f in failures_json]
    
    # Get per-task durations (scan for duration keys)
    duration_keys = r.keys(f"{DURATIONS_KEY}:*")
    metrics["avg_duration_per_task"] = {}
    for key in duration_keys:
        task_name = key.replace(f"{DURATIONS_KEY}:", "")
        durations_json = r.lrange(key, 0, _MAX_DURATION_HISTORY - 1)
        durations = [float(d) for d in durations_json]
        if durations:
            metrics["task_durations"][task_name] = durations
            metrics["avg_duration_per_task"][task_name] = sum(durations) / len(durations)
        else:
            metrics["avg_duration_per_task"][task_name] = 0.0
    
    return metrics


def get_recent_failures(limit: int = 10) -> list[dict]:
    """Get recent task failures from Redis."""
    r = _get_redis()
    failures_json = r.lrange(FAILURES_KEY, 0, limit - 1)
    return [json.loads(f) for f in failures_json]


def check_health() -> dict:
    """Health check for task processing system."""
    metrics = get_task_metrics()
    
    # Determine health status
    status = "healthy"
    issues = []
    
    if metrics["total_tasks"] > 10 and metrics["failure_rate"] > 0.2:
        status = "degraded"
        issues.append(f"High failure rate: {metrics['failure_rate']:.1%}")
    
    if metrics["total_tasks"] > 5 and metrics["avg_duration_seconds"] > 300:
        status = "degraded"
        issues.append(f"High average duration: {metrics['avg_duration_seconds']:.0f}s")
    
    # Check for recent failures
    recent_failures = get_recent_failures(5)
    if recent_failures:
        last_failure = recent_failures[-1]
        issues.append(f"Last failure: {last_failure['task_name']} - {last_failure['error']}")
    
    return {
        "status": status,
        "issues": issues,
        "metrics": metrics,
    }


# Celery signal handlers
@signals.task_prerun.connect
def task_prerun_handler(task_id: str, task: Task, *args, **kwargs) -> None:
    """Record task start time."""
    task._start_time = time.time()
    logger.info(f"Task started: {task.name} [{task_id}]")


@signals.task_postrun.connect
def task_postrun_handler(task_id: str, task: Task, retval: Any, state: str, *args, **kwargs) -> None:
    """Record task completion."""
    duration = time.time() - getattr(task, "_start_time", time.time())
    success = state == "SUCCESS"
    error = None if success else str(retval) if retval else "Unknown error"
    
    _record_metric(task.name, duration, success, error)

    try:
        from backend.app.metrics import CELERY_TASK_EXECUTED, CELERY_TASK_DURATION
        status_label = "success" if success else "failure"
        CELERY_TASK_EXECUTED.labels(task_name=task.name or "unknown", status=status_label).inc()
        CELERY_TASK_DURATION.labels(task_name=task.name or "unknown").observe(duration)
    except Exception as exc:
        logger.debug("Could not record prometheus celery metrics: %s", exc)
    
    if success:
        logger.info(f"Task completed: {task.name} [{task_id}] in {duration:.2f}s")
    else:
        logger.error(f"Task failed: {task.name} [{task_id}] after {duration:.2f}s - {error}")


@signals.task_retry.connect
def task_retry_handler(request, reason: str, einfo, *args, **kwargs) -> None:
    """Record task retry."""
    r = _get_redis()
    r.hincrby(METRICS_KEY, "retried_tasks", 1)
    logger.warning(f"Task retry: {request.task.name} [{request.id}] - {reason}")


@signals.task_failure.connect
def task_failure_handler(task_id: str, exception: Exception, traceback: str, *args, **kwargs) -> None:
    """Log detailed task failure information."""
    logger.error(
        f"Task failure: {task_id} - {type(exception).__name__}: {exception}\n{traceback}"
    )


# Decorator for manual task monitoring (for tasks not using Celery signals)
def monitor_task(task_name: Optional[str] = None) -> Callable:
    """
    Decorator to monitor any function as a task.
    Usage:
        @monitor_task("my_custom_task")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = task_name or f"{func.__module__}.{func.__name__}"
            start = time.time()
            error = None
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start
                _record_metric(name, duration, error is None, error)
        return wrapper
    return decorator


# Alerting helpers
def check_and_alert(threshold_failure_rate: float = 0.2, threshold_avg_duration: float = 300) -> list[str]:
    """
    Check metrics and return alert messages if thresholds exceeded.
    In production, integrate with alerting system (PagerDuty, Slack, email, etc.)
    """
    alerts = []
    metrics = get_task_metrics()
    
    if metrics["total_tasks"] > 10 and metrics["failure_rate"] > threshold_failure_rate:
        alerts.append(
            f"ALERT: High task failure rate {metrics['failure_rate']:.1%} "
            f"(threshold: {threshold_failure_rate:.1%})"
        )
    
    if metrics["total_tasks"] > 5 and metrics["avg_duration_seconds"] > threshold_avg_duration:
        alerts.append(
            f"ALERT: High average task duration {metrics['avg_duration_seconds']:.0f}s "
            f"(threshold: {threshold_avg_duration}s)"
        )
    
    # Check specific critical tasks
    critical_tasks = [
        "backend.app.workers.train.train_house_models_task",
        "backend.app.workers.fixtures.scrape_all_fixtures_task",
        "backend.app.workers.predict.generate_daily_predictions_task",
        "backend.app.workers.data_sync.sync_all_leagues_task",
    ]
    
    for task_name in critical_tasks:
        if task_name in metrics["task_durations"]:
            durations = metrics["task_durations"][task_name]
            if durations:
                recent_avg = sum(durations[-5:]) / min(len(durations), 5)
                # Alert if recent average is 2x the overall average
                if len(durations) > 5 and recent_avg > metrics["avg_duration_seconds"] * 2:
                    alerts.append(
                        f"ALERT: Task {task_name} recent avg duration {recent_avg:.0f}s "
                        f"is 2x overall avg {metrics['avg_duration_seconds']:.0f}s"
                    )
    
    return alerts


# Initialize signal handlers when module is imported
def init_monitoring() -> None:
    """Initialize monitoring - call during app startup."""
    logger.info("Celery task monitoring initialized")
    # Signals are automatically connected via decorators above


# Auto-initialize
init_monitoring()