"""Linear Integration API Router for ProphitBet.

Exposes endpoints for dispatching predictions, scores, dev tasks,
status checks, and receiving Linear webhooks.
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import get_db
from backend.app.services.alert_service import AlertService
from backend.app.services.linear_service import LinearService

logger = logging.getLogger(__name__)

router = APIRouter()


class CreateTaskRequest(BaseModel):
    title: str
    description: str
    priority: Optional[int] = 2


class PostUpdateRequest(BaseModel):
    body: str


@router.get("/status")
async def get_linear_status():
    """Verify Linear API key and fetch project status."""
    team_id = LinearService.get_team_id()
    if not team_id:
        raise HTTPException(status_code=400, detail="Could not connect to Linear API or fetch team.")

    project_id = LinearService.get_or_create_project()
    return {
        "status": "connected",
        "team_id": team_id,
        "project_id": project_id,
        "project_name": "ProphitBet Soccer Predictor",
    }


@router.post("/dispatch-alerts")
async def dispatch_linear_alerts(
    min_ev_pct: float = 5.0,
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
):
    """Scan database for high +EV predictions and post them as Linear issues."""
    alerts = await AlertService.get_high_value_alerts(db, min_ev_pct=min_ev_pct, limit=limit)
    created_issues = []

    for alert in alerts:
        issue = LinearService.create_prediction_issue(alert)
        if issue:
            created_issues.append(issue)

    return {
        "dispatched": len(created_issues),
        "issues": created_issues,
    }


@router.post("/create-task")
async def create_linear_task(req: CreateTaskRequest):
    """Create a dev or coding task in Linear."""
    issue = LinearService.create_dev_task(title=req.title, description=req.description, priority=req.priority)
    if not issue:
        raise HTTPException(status_code=500, detail="Failed to create issue in Linear")
    return {"status": "created", "issue": issue}


@router.post("/webhook")
async def handle_linear_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive incoming webhooks from Linear for autonomous agent tracking."""
    try:
        payload = await request.json()
        action = payload.get("action")
        type_ = payload.get("type")
        data = payload.get("data", {})

        logger.info(f"Received Linear Webhook: type={type_}, action={action}, data_id={data.get('id')}")

        # Handle comment or issue updates tagged for AI / Antigravity
        if type_ == "Issue" and action in ("create", "update"):
            title = data.get("title", "")
            desc = data.get("description", "")
            if "[AGY]" in title or "Antigravity" in desc:
                logger.info(f"Linear issue tagged for Antigravity: {title}")

        return {"status": "received", "action": action, "type": type_}
    except Exception as e:
        logger.error(f"Error processing Linear webhook: {e}")
        return {"status": "error", "detail": str(e)}
