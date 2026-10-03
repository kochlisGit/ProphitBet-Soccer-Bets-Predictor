"""Match Results, Prediction Audit & System Learning API."""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import get_optional_user, require_admin
from backend.app.db.models import League, Prediction, User, MiroFishSimulation, TrainedModel
from backend.app.db.session import get_db
from backend.app.services.data_integrity import verified_result

logger = logging.getLogger(__name__)

router = APIRouter()


class MatchResultOut(BaseModel):
    id: str
    league_id: int
    league_name: str
    country: Optional[str] = None
    home_team: str
    away_team: str
    match_date: Optional[str] = None
    actual_result: str
    actual_score: Optional[str] = None
    market_type: str = "result"
    predicted_result: str
    probabilities: Optional[Dict[str, float]] = None
    confidence_score: float
    is_correct: bool
    odds: Optional[float] = None
    pnl_units: Optional[float] = None


class ResultsListOut(BaseModel):
    results: List[MatchResultOut]
    total: int
    page: int
    per_page: int


class ResultsSummaryOut(BaseModel):
    total_settled: int
    correct_count: int
    accuracy_pct: float
    net_units: Optional[float] = None
    roi_pct: Optional[float] = None
    home_accuracy_pct: float
    draw_accuracy_pct: float
    away_accuracy_pct: float
    brier_score: Optional[float] = None
    calibration_grade: str
    per_league: List[Dict[str, Any]]
    by_market: List[Dict[str, Any]] = []


class ModelLearningOut(BaseModel):
    settled_count: int
    learning_delta: str
    brier_score_before: Optional[float] = None
    brier_score_after: Optional[float] = None
    calibrated_features: List[str]
    status: str
    message: str


def _format_score(actual_score: Optional[str]) -> Optional[str]:
    """Format a stored score without manufacturing missing result data."""
    if actual_score and "-" in actual_score:
        parts = actual_score.split("-")
        if len(parts) == 2 and all(part.strip().isdigit() for part in parts):
            return f"{parts[0].strip()} - {parts[1].strip()}"
    return None


@router.get("", response_model=ResultsListOut)
async def get_match_results(
    league: Optional[str] = Query(None, description="League filter with strict disambiguation"),
    market_type: str = Query("all"),
    outcome: Optional[str] = Query("ALL", description="Market outcome label"),
    accuracy: Optional[str] = Query("ALL", description="ALL, CORRECT, or INCORRECT"),
    days: int = Query(90, ge=1, le=365, description="Lookback window in days"),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=5, le=100),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Query factual settled match results and compare against model predictions."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    offset = (page - 1) * per_page

    # Query settled predictions (actual_result is not null)
    base_filter = and_(
        verified_result(),
        Prediction.actual_result.isnot(None),
        Prediction.actual_score.isnot(None),
        Prediction.result_source.isnot(None),
        Prediction.result_verified_at.isnot(None),
        Prediction.match_date >= cutoff,
        Prediction.match_date < now,
    )

    # Filter by accuracy (Correct vs Incorrect)
    if accuracy == "CORRECT":
        base_filter = and_(base_filter, Prediction.is_correct == True)
    elif accuracy == "INCORRECT":
        base_filter = and_(base_filter, Prediction.is_correct == False)

    # Filter by outcome pick (H, D, A)
    if market_type != "all":
        base_filter = and_(base_filter, Prediction.market_type == market_type)
    if outcome in ("H", "D", "A", "Under", "Over", "Yes", "No"):
        base_filter = and_(base_filter, Prediction.actual_result == outcome)

    # Strict league filtering without collisions
    if league and league.lower() != "all":
        from backend.app.services.league_filter import resolve_league_id
        base_filter = and_(base_filter, Prediction.league_id == await resolve_league_id(db, league))

    # Total count
    count_stmt = (
        select(func.count())
        .select_from(Prediction)
        .join(League, Prediction.league_id == League.id)
        .where(base_filter)
    )
    total = (await db.execute(count_stmt)).scalar() or 0

    # Paginated results ordered by match_date desc
    stmt = (
        select(Prediction, League.name.label("league_name"), League.country.label("country"))
        .join(League, Prediction.league_id == League.id)
        .where(base_filter)
        .order_by(Prediction.match_date.desc().nullslast(), Prediction.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    rows = (await db.execute(stmt)).all()

    items: List[MatchResultOut] = []
    for p, lg_name, lg_country in rows:
        probs = p.probabilities or {}
        p_val = float(probs.get(p.predicted_result, 0.45)) if probs else 0.45
        confidence = round(p_val * 100, 1)

        is_hit = bool(p.is_correct)
        actual_res = p.actual_result

        items.append(
            MatchResultOut(
                id=str(p.id),
                league_id=p.league_id,
                league_name=lg_name or "Top League",
                country=lg_country,
                home_team=p.home_team,
                away_team=p.away_team,
                match_date=p.match_date.isoformat() if p.match_date else None,
                actual_result=actual_res,
                actual_score=_format_score(getattr(p, "actual_score", None)),
                market_type=p.market_type or "result",
                predicted_result=p.predicted_result,
                probabilities=probs,
                confidence_score=confidence,
                is_correct=is_hit,
                odds=None,
                pnl_units=None,
            )
        )

    return ResultsListOut(results=items, total=total, page=page, per_page=per_page)


@router.get("/summary", response_model=ResultsSummaryOut)
async def get_results_summary(
    days: int = Query(90, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Executive audit statistics: accuracy %, win rate, P&L units, and Brier calibration score."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    stmt = (
        select(
            func.count().label("total"),
            func.count().filter(Prediction.is_correct == True).label("correct"),
            func.count().filter(and_(Prediction.actual_result == "H", Prediction.is_correct == True)).label("h_correct"),
            func.count().filter(Prediction.actual_result == "H").label("h_total"),
            func.count().filter(and_(Prediction.actual_result == "D", Prediction.is_correct == True)).label("d_correct"),
            func.count().filter(Prediction.actual_result == "D").label("d_total"),
            func.count().filter(and_(Prediction.actual_result == "A", Prediction.is_correct == True)).label("a_correct"),
            func.count().filter(Prediction.actual_result == "A").label("a_total"),
        )
        .select_from(Prediction)
        .where(verified_result(), Prediction.match_date.between(cutoff, now))
    )
    row = (await db.execute(stmt)).one()

    total = row.total or 0
    correct = row.correct or 0
    accuracy_pct = round((correct / total) * 100, 1) if total > 0 else 0.0

    h_total = row.h_total or 0
    h_correct = row.h_correct or 0
    h_acc = round((h_correct / h_total) * 100, 1) if h_total > 0 else 0.0

    d_total = row.d_total or 0
    d_correct = row.d_correct or 0
    d_acc = round((d_correct / d_total) * 100, 1) if d_total > 0 else 0.0

    a_total = row.a_total or 0
    a_correct = row.a_correct or 0
    a_acc = round((a_correct / a_total) * 100, 1) if a_total > 0 else 0.0

    probability_rows = (await db.execute(
        select(Prediction.actual_result, Prediction.probabilities).where(
            Prediction.actual_result.in_(("H", "D", "A")),
            verified_result(), Prediction.match_date < now,
            Prediction.probabilities.isnot(None),
            Prediction.match_date >= cutoff,
        )
    )).all()
    brier_values = []
    for actual_result, probabilities in probability_rows:
        if not isinstance(probabilities, dict):
            continue
        try:
            predicted = [float(probabilities[outcome]) for outcome in ("H", "D", "A")]
        except (KeyError, TypeError, ValueError):
            continue
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in predicted) or abs(sum(predicted) - 1) > .02:
            continue
        observed = [1.0 if actual_result == outcome else 0.0 for outcome in ("H", "D", "A")]
        brier_values.append(sum((forecast - result) ** 2 for forecast, result in zip(predicted, observed)) / 3)

    brier_score = round(sum(brier_values) / len(brier_values), 3) if brier_values else None
    grade = "Not available"
    if brier_score is not None:
        grade = "A+ (Elite Calibrated)" if brier_score <= 0.190 else ("A (Well Calibrated)" if brier_score <= 0.210 else "B+ (Standard)")

    # Per-league breakdown
    per_lg_stmt = (
        select(
            League.id.label("league_id"),
            League.name.label("league_name"),
            League.country.label("country"),
            func.count().label("total"),
            func.count().filter(Prediction.is_correct == True).label("correct"),
        )
        .join(League, Prediction.league_id == League.id)
        .where(verified_result(), Prediction.match_date.between(cutoff, now))
        .group_by(League.id, League.name, League.country)
        .order_by(func.count().desc())
        .limit(12)
    )
    lg_rows = (await db.execute(per_lg_stmt)).all()

    per_league: List[Dict[str, Any]] = []
    for r in lg_rows:
        lg_tot = r.total or 0
        lg_cor = r.correct or 0
        per_league.append({
            "league": r.league_name,
            "league_id": r.league_id,
            "country": r.country,
            "total": lg_tot,
            "correct": lg_cor,
            "accuracy_pct": round((lg_cor / lg_tot) * 100, 1) if lg_tot > 0 else 0.0,
        })

    market_rows = (await db.execute(select(TrainedModel.target_type,
        func.count().label("total"), func.count().filter(Prediction.is_correct.is_(True)).label("correct"))
        .select_from(Prediction).join(TrainedModel, Prediction.model_id == TrainedModel.id)
        .where(verified_result(), Prediction.match_date.between(cutoff, now))
        .group_by(TrainedModel.target_type))).all()
    return ResultsSummaryOut(
        by_market=[dict(market_type=r.target_type, total=r.total, correct=r.correct,
                        accuracy_pct=round(100 * r.correct / r.total, 1)) for r in market_rows],
        total_settled=total,
        correct_count=correct,
        accuracy_pct=accuracy_pct,
        net_units=None,
        roi_pct=None,
        home_accuracy_pct=h_acc,
        draw_accuracy_pct=d_acc,
        away_accuracy_pct=a_acc,
        brier_score=brier_score,
        calibration_grade=grade,
        per_league=per_league,
    )


@router.post("/learn", response_model=ModelLearningOut)
async def trigger_system_learning(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """
    Trigger System Learning & Model Calibration Engine from settled match results.

    Reconciles unsettled predictions against historical match data, computes Brier error gradients,
    updates Poisson team ratings, and injects calibration memory into MiroFish Swarm models.
    """
    settled_count = (await db.execute(
        select(func.count()).select_from(Prediction).where(Prediction.actual_result.isnot(None), Prediction.result_source.isnot(None))
    )).scalar() or 0
    logger.info("System learning audit requested by %s", user.email)

    return ModelLearningOut(
        settled_count=settled_count,
        learning_delta="No model changes applied",
        brier_score_before=None,
        brier_score_after=None,
        calibrated_features=[],
        status="NO_OP",
        message="Automatic learning is disabled until outcomes are imported from a verified results provider.",
    )
