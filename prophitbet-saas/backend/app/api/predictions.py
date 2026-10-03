from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func, and_, or_, case
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import get_current_user, get_optional_user, require_plan
from backend.app.db.models import League, Prediction, User
from backend.app.db.session import get_db
from backend.app.services.data_integrity import publishable_prediction, verified_result

router = APIRouter()

FREE_LEAGUE_LIMIT = 5


class PredictionOut(BaseModel):
    id: str
    league_id: int
    league_name: str
    country: Optional[str] = None
    home_team: str
    away_team: str
    match_date: Optional[str] = None
    market_type: str = "result"
    predicted_result: str
    probabilities: Optional[dict] = None
    actual_result: Optional[str] = None
    is_correct: Optional[bool] = None
    created_at: str


class PredictionListOut(BaseModel):
    predictions: list[PredictionOut]
    total: int
    page: int
    per_page: int


def _plan_of(user: Optional[User]) -> str:
    if user is None:
        return "free"
    if getattr(user, "is_admin", False):
        return "elite"
    if user.subscription and user.subscription.status == "active":
        return user.subscription.plan
    return "free"


@router.get("/today", response_model=PredictionListOut)
async def get_today_predictions(
    market_type: str = Query("all", description="Prediction market, or all"),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Predictions created in the last 7 days (dashboard feed; not only calendar 'today')."""
    plan = _plan_of(user)
    
    now = datetime.now(timezone.utc)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    recent_start = now - timedelta(days=7)

    stmt = (
        select(Prediction, League.name.label("league_name"), League.country.label("country"))
        .join(League, Prediction.league_id == League.id)
        .where(
            publishable_prediction(now),
            or_(
                and_(Prediction.match_date >= start_of_today, Prediction.actual_result.is_(None)),
                Prediction.created_at >= recent_start,
            )
        )
        .order_by(
            case((and_(Prediction.match_date >= start_of_today, Prediction.actual_result.is_(None)), 0), else_=1),
            Prediction.match_date.asc().nullslast(),
            Prediction.created_at.desc(),
        )
        .limit(250)
    )

    if plan == "free":
        top_league_ids = (
            select(Prediction.league_id)
            .distinct()
            .limit(FREE_LEAGUE_LIMIT)
        )
        stmt = stmt.where(Prediction.league_id.in_(top_league_ids))

    if market_type != "all":
        stmt = stmt.where(Prediction.market_type == market_type)
    result = await db.execute(stmt)
    rows = result.all()

    preds = []
    for pred, league_name, country in rows:
        out = PredictionOut(
            id=str(pred.id),
            league_id=pred.league_id,
            league_name=league_name,
            country=country,
            home_team=pred.home_team,
            away_team=pred.away_team,
            match_date=pred.match_date.isoformat() if pred.match_date else None,
            market_type=pred.market_type or "result",
            predicted_result=pred.predicted_result,
            probabilities=pred.probabilities if plan != "free" else None,
            actual_result=pred.actual_result,
            is_correct=pred.is_correct,
            created_at=pred.created_at.isoformat(),
        )
        preds.append(out)

    return PredictionListOut(predictions=preds, total=len(preds), page=1, per_page=len(preds))


@router.get("/league/{league_id}", response_model=PredictionListOut)
async def get_league_predictions(
    league_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=5, le=100),
    market_type: str = Query("all", description="Prediction market, or all"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    plan = _plan_of(user)
    
    offset = (page - 1) * per_page

    count_result = await db.execute(
        select(func.count()).select_from(Prediction).where(Prediction.league_id == league_id, publishable_prediction(),
            True if market_type == "all" else Prediction.market_type == market_type)
    )
    total = count_result.scalar() or 0

    now = datetime.now(timezone.utc)
    upcoming_start = now - timedelta(hours=36)

    stmt = (
        select(Prediction, League.name.label("league_name"), League.country.label("country"))
        .join(League, Prediction.league_id == League.id)
        .where(Prediction.league_id == league_id, publishable_prediction(now))
        .order_by(
            case((and_(Prediction.match_date >= upcoming_start, Prediction.actual_result.is_(None)), 0), else_=1),
            Prediction.match_date.asc().nullslast(),
            Prediction.created_at.desc(),
        )
        .offset(offset)
        .limit(per_page)
    )
    if market_type != "all":
        stmt = stmt.where(Prediction.market_type == market_type)
    result = await db.execute(stmt)
    rows = result.all()

    preds = []
    for pred, league_name, country in rows:
        preds.append(PredictionOut(
            id=str(pred.id),
            league_id=pred.league_id,
            league_name=league_name,
            country=country,
            home_team=pred.home_team,
            away_team=pred.away_team,
            match_date=pred.match_date.isoformat() if pred.match_date else None,
            market_type=pred.market_type or "result",
            predicted_result=pred.predicted_result,
            probabilities=pred.probabilities if plan != "free" else None,
            actual_result=pred.actual_result,
            is_correct=pred.is_correct,
            created_at=pred.created_at.isoformat(),
        ))

    return PredictionListOut(predictions=preds, total=total, page=page, per_page=per_page)


@router.get("/history", response_model=PredictionListOut)
async def get_prediction_history(
    days: int = Query(7, ge=1, le=365),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=5, le=100),
    settled_only: bool = Query(
        False,
        description="If true, only predictions with an actual result (for accuracy views).",
    ),
    league_id: Optional[int] = Query(None, description="Optional league ID to filter predictions."),
    league: Optional[str] = Query(None, description="Optional league name to filter predictions."),
    market_type: str = Query("all", description="Prediction market, or all"),
    group_matches: bool = Query(False, description="Paginate complete matches rather than individual market rows."),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    plan = _plan_of(user)
    if plan == "free" and days > 7:
        days = 7
    elif plan == "pro" and days > 90:
        days = 90

    now = datetime.now(timezone.utc)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = now - timedelta(days=days)
    offset = (page - 1) * per_page

    # Include upcoming picks (match_date >= start_of_today) or recently created predictions
    if settled_only:
        base_filter = and_(verified_result(), Prediction.match_date >= cutoff)
    else:
        base_filter = or_(
            and_(Prediction.match_date >= start_of_today, Prediction.actual_result.is_(None)),
            Prediction.created_at >= cutoff,
        )

    base_filter = and_(base_filter, publishable_prediction(now),
        True if market_type == "all" else Prediction.market_type == market_type)
    if league_id is not None:
        base_filter = and_(base_filter, Prediction.league_id == league_id)
    elif league and league.lower() != "all":
        from backend.app.services.league_filter import resolve_league_id
        base_filter = and_(base_filter, Prediction.league_id == await resolve_league_id(db, league))

    if plan == "free":
        top_league_ids = (
            select(Prediction.league_id)
            .distinct()
            .limit(FREE_LEAGUE_LIMIT)
        )
        base_filter = and_(base_filter, Prediction.league_id.in_(top_league_ids))

    count_result = await db.execute(
        select(func.count())
        .select_from(Prediction)
        .join(League, Prediction.league_id == League.id)
        .where(base_filter)
    )
    total = count_result.scalar() or 0

    stmt = (
        select(Prediction, League.name.label("league_name"), League.country.label("country"))
        .join(League, Prediction.league_id == League.id)
        .where(base_filter)
        .order_by(
            case((and_(Prediction.match_date >= start_of_today, Prediction.actual_result.is_(None)), 0), else_=1),
            Prediction.match_date.asc().nullslast(),
            Prediction.created_at.desc(),
        )
        .offset(offset)
        .limit(per_page)
    )
    if market_type != "all":
        stmt = stmt.where(Prediction.market_type == market_type)
    result = await db.execute(stmt)
    rows = result.all()

    if group_matches:
        # Page by fixture identity so a match's markets never spill onto another page.
        keys = [Prediction.league_id, func.lower(func.trim(Prediction.home_team)),
                func.lower(func.trim(Prediction.away_team)), Prediction.match_date,
                case((Prediction.match_date.is_(None), Prediction.id), else_=None)]
        matches = (
            select(*(key.label(f"key_{i}") for i, key in enumerate(keys)),
                   func.min(case((and_(Prediction.match_date >= start_of_today,
                                       Prediction.actual_result.is_(None)), 0), else_=1)).label("priority"))
            .join(League, Prediction.league_id == League.id)
            .where(base_filter).group_by(*keys)
        ).subquery()
        total = (await db.execute(select(func.count()).select_from(matches))).scalar() or 0
        selected = (select(matches).order_by(matches.c.priority, matches.c.key_3.asc().nullslast(),
                    matches.c.key_0, matches.c.key_1, matches.c.key_2, matches.c.key_4)
                    .offset(offset).limit(per_page).subquery())
        grouped_stmt = (select(Prediction, League.name, League.country)
                        .join(League, Prediction.league_id == League.id)
                        .join(selected, and_(*(key.is_not_distinct_from(selected.c[f"key_{i}"])
                                               for i, key in enumerate(keys))))
                        .where(base_filter)
                        .order_by(Prediction.match_date.asc().nullslast(), Prediction.created_at.desc(), Prediction.id))
        rows = (await db.execute(grouped_stmt)).all()

    preds = [
        PredictionOut(
            id=str(p.id),
            league_id=p.league_id,
            league_name=ln,
            country=country,
            home_team=p.home_team,
            away_team=p.away_team,
            match_date=p.match_date.isoformat() if p.match_date else None,
            market_type=p.market_type or "result",
            predicted_result=p.predicted_result,
            probabilities=p.probabilities if plan != "free" else None,
            actual_result=p.actual_result,
            is_correct=p.is_correct,
            created_at=p.created_at.isoformat(),
        )
        for p, ln, country in rows
    ]
    return PredictionListOut(predictions=preds, total=total, page=page, per_page=per_page)


@router.get("/accuracy/stats")
async def get_accuracy_stats(
    days: int = Query(30, ge=1, le=365),
    market_type: str = Query("all", description="Prediction market, or all"),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Platform-wide prediction accuracy statistics."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    resolved = await db.execute(
        select(
            func.count().label("total"),
            func.count().filter(Prediction.is_correct == True).label("correct"),
        )
        .select_from(Prediction)
        .where(verified_result(), Prediction.match_date >= cutoff, Prediction.match_date < now,
               True if market_type == "all" else Prediction.market_type == market_type)
    )
    row = resolved.one()
    total = row.total or 0
    correct = row.correct or 0
    accuracy = round(correct / total, 4) if total > 0 else 0

    per_league = await db.execute(
        select(
            League.name.label("league_name"),
            func.count().label("total"),
            func.count().filter(Prediction.is_correct == True).label("correct"),
        )
        .join(League, Prediction.league_id == League.id)
        .where(verified_result(), Prediction.match_date >= cutoff, Prediction.match_date < now,
               True if market_type == "all" else Prediction.market_type == market_type)
        .group_by(League.name)
        .order_by(func.count().desc())
        .limit(20)
    )
    league_stats = [
        {
            "league": r.league_name,
            "total": r.total,
            "correct": r.correct,
            "accuracy": round(r.correct / r.total, 4) if r.total > 0 else 0,
        }
        for r in per_league.all()
    ]

    return {
        "days": days,
        "total_predictions": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "by_league": league_stats,
    }


@router.get("/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    market_type: str = Query("all", description="Prediction market, or all"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from uuid import UUID
    try:
        pid = UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction ID")

    result = await db.execute(
        select(Prediction, League.name.label("league_name"))
        .join(League, Prediction.league_id == League.id)
        .where(Prediction.id == pid, publishable_prediction())
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    pred, league_name = row
    plan = _plan_of(user)
    return PredictionOut(
        id=str(pred.id),
        league_id=pred.league_id,
        league_name=league_name,
        home_team=pred.home_team,
        away_team=pred.away_team,
        match_date=pred.match_date.isoformat() if pred.match_date else None,
        market_type=pred.market_type or "result",
            predicted_result=pred.predicted_result,
        probabilities=pred.probabilities if plan != "free" else None,
        actual_result=pred.actual_result,
        is_correct=pred.is_correct,
        created_at=pred.created_at.isoformat(),
    )
