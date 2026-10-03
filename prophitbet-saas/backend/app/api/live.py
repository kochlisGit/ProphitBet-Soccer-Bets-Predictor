"""Live Match Watch & Play-by-Play Commentary API router."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import case, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import get_optional_user
from backend.app.db.models import Fixture, League, Prediction, User
from backend.app.db.session import get_db
from backend.app.services.data_integrity import publishable_prediction
from backend.app.services.live_match_service import LiveMatchEngine

logger = logging.getLogger(__name__)

router = APIRouter()


class LiveMatchItem(BaseModel):
    id: str
    home_team: str
    away_team: str
    league_id: int
    league_name: str
    country: Optional[str] = None
    match_date: Optional[datetime] = None
    predicted_result: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    status: str = "UPCOMING"  # No live-event provider is configured.
    current_minute: Optional[int] = None
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    odds_1: Optional[float] = None
    odds_x: Optional[float] = None
    odds_2: Optional[float] = None


@router.get("/matches", response_model=List[LiveMatchItem])
async def get_live_matches(
    league: Optional[str] = Query(None, description="Filter by league name"),
    status: Optional[str] = Query(None, description="LIVE, UPCOMING, or ALL"),
    limit: int = Query(60, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Retrieve list of matches available for Live Match Watch and Commentary."""
    stmt = (
        select(Prediction, League.name.label("league_name"), League.country.label("league_country"))
        .join(League, Prediction.league_id == League.id)
        .where(publishable_prediction())
        .order_by(case((Prediction.actual_result.is_(None), 0), else_=1), Prediction.match_date.asc())
        .limit(limit)
    )

    if status == "LIVE":
        return []  # A scheduled kickoff is not proof a match is live.
    if status == "UPCOMING":
        stmt = stmt.where(Prediction.actual_result.is_(None))
    elif status == "FINISHED":
        stmt = stmt.where(Prediction.actual_result.isnot(None))

    if league and league != "ALL":
        norm = league.lower().replace("-", " ").replace("_", " ").strip()
        stmt = stmt.where(
            or_(
                League.name.ilike(f"%{norm}%"),
                League.country.ilike(f"%{norm}%"),
            )
        )

    result = await db.execute(stmt)
    rows = result.all()

    now = datetime.now(timezone.utc)
    match_list: List[LiveMatchItem] = []

    for pred, lg_name, lg_country in rows:
        probs = pred.probabilities or {}

        # Determine live status and minute
        m_date = pred.match_date
        current_minute = None
        match_status = "UPCOMING"
        score_h = None
        score_a = None

        if m_date:
            # If date has no tzinfo, assume UTC
            if m_date.tzinfo is None:
                m_date = m_date.replace(tzinfo=timezone.utc)

            if pred.actual_score and pred.actual_result:
                match_status = "FINISHED"
                parts = pred.actual_score.replace(" ", "").split("-")
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    score_h, score_a = int(parts[0]), int(parts[1])

        match_list.append(
            LiveMatchItem(
                id=str(pred.id),
                home_team=pred.home_team,
                away_team=pred.away_team,
                league_id=pred.league_id,
                league_name=lg_name or "Premier League",
                country=lg_country,
                match_date=pred.match_date,
                predicted_result=pred.predicted_result,
                probabilities=probs or None,
                status=match_status,
                current_minute=current_minute,
                score_home=score_h,
                score_away=score_a,
                odds_1=None,
                odds_x=None,
                odds_2=None,
            )
        )

    return match_list


@router.get("/matches/{match_id}/timeline")
async def get_match_timeline(
    match_id: str,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Retrieve full 90-minute tactical simulation timeline with play-by-play commentary."""
    raise HTTPException(
        status_code=501,
        detail="Live timeline unavailable: no verified live-event provider is configured",
    )
    # Look up prediction
    stmt = (
        select(Prediction, League.name.label("league_name"), League.country.label("league_country"))
        .join(League, Prediction.league_id == League.id)
        .where(Prediction.id == match_id, publishable_prediction())
    )
    res = await db.execute(stmt)
    row = res.first()

    if not row:
        # Fallback search by Fixture
        f_stmt = select(Fixture, League.name.label("league_name")).outerjoin(League, Fixture.league_id == League.id).where(Fixture.id == match_id, Fixture.is_current.is_(True))
        f_res = await db.execute(f_stmt)
        f_row = f_res.first()
        if not f_row:
            raise HTTPException(status_code=404, detail="Match not found")

        fix, lg_name = f_row
        home_team = fix.home_team
        away_team = fix.away_team
        league_name = lg_name or "Premier League"
        prob_h = 0.45
        prob_d = 0.28
        prob_a = 0.27
    else:
        pred, lg_name, lg_country = row
        home_team = pred.home_team
        away_team = pred.away_team
        league_name = lg_name or "Premier League"
        probs = pred.probabilities or {}
        prob_h = float(probs.get("H", 0.45))
        prob_d = float(probs.get("D", 0.28))
        prob_a = float(probs.get("A", 0.27))

    # Calculate projected goals based on probability weights
    projected_h = 0.9 + prob_h * 1.5
    projected_a = 0.7 + prob_a * 1.3

    timeline_data = LiveMatchEngine.generate_match_timeline(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        league_name=league_name,
        prob_home=prob_h,
        prob_draw=prob_d,
        prob_away=prob_a,
        projected_home_goals=projected_h,
        projected_away_goals=projected_a,
    )

    return timeline_data
