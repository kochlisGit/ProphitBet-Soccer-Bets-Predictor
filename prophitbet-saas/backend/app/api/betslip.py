"""User Bet Slip & Personal Bankroll Journal API Routes.

Allows authenticated users to log personal bets, track closing line value,
monitor empirical ROI, and auto-settle outcomes against live match results.
"""

import uuid
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
import math

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.auth.dependencies import get_current_user
from backend.app.db.models import User, UserBetSlip, Prediction, League, SABookmakerOdds, Fixture
from backend.app.db.session import get_db
from backend.app.services.data_integrity import verified_result
from backend.app.services.betting_integrity import MARKETS, normalize_selection, settlement, settle_legs, money, observed_quote

router = APIRouter()


def selection_won(selection: str, actual: str, score: str):
    """None means unsupported/unverifiable; leave that journal entry pending."""
    result = settlement(selection, actual, score)
    return result == "WON" if result in ("WON", "LOST") else None


class BetLegRequest(BaseModel):
    prediction_id: Optional[uuid.UUID] = None
    quote_id: Optional[uuid.UUID] = None
    match_title: str = Field(min_length=1, max_length=250)
    match_date: Optional[datetime] = None
    selection: str
    odds_taken: float = Field(gt=1, le=10000, allow_inf_nan=False)
    bookmaker: Literal["HOLLYWOODBETS", "BETWAY"]

    @field_validator("selection")
    @classmethod
    def canonical_selection(cls, value):
        return normalize_selection(value)


class CreateBetSlipRequest(BaseModel):
    prediction_id: Optional[uuid.UUID] = None
    match_title: str = Field(default="Accumulator", min_length=1, max_length=250)
    league_name: Optional[str] = Field(default="Top League", max_length=100)
    selection: str = "H"
    odds_taken: Optional[float] = Field(default=None, gt=1.0, le=1000000, allow_inf_nan=False)
    stake_amount: float = Field(default=100.0, ge=0.01, le=1000000, allow_inf_nan=False)
    closing_odds: Optional[float] = Field(default=None, gt=1, le=1000000, allow_inf_nan=False)
    bookmaker: Optional[Literal["HOLLYWOODBETS", "BETWAY"]] = None
    legs: Optional[List[BetLegRequest]] = Field(default=None, min_length=1, max_length=20)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_slip(self):
        if not self.legs:
            if self.odds_taken is None:
                raise ValueError("Enter the odds from your bookmaker receipt")
            self.selection = normalize_selection(self.selection)
        else:
            if len({leg.bookmaker for leg in self.legs}) != 1:
                raise ValueError("An accumulator must use one bookmaker")
            keys = [leg.match_title.strip().casefold() for leg in self.legs]
            if len(set(keys)) != len(keys):
                raise ValueError("Same-event combinations require a bookmaker-supplied combined price")
            if math.prod(leg.odds_taken for leg in self.legs) > 1000000:
                raise ValueError("Combined odds exceed the journal limit")
        return self


class UpdateBetSlipRequest(BaseModel):
    status: Optional[Literal["PENDING", "WON", "LOST", "VOID"]] = None
    closing_odds: Optional[float] = Field(default=None, gt=1, le=1000000, allow_inf_nan=False)
    notes: Optional[str] = None

    model_config = {"extra": "forbid"}


@router.get("")
async def get_user_bet_slips(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Retrieve personal bet journal, bankroll metrics, and trade history."""
    stmt = (
        select(UserBetSlip)
        .where(UserBetSlip.user_id == user.id)
        .order_by(UserBetSlip.created_at.desc())
    )
    if status and status.upper() != "ALL":
        stmt = stmt.where(UserBetSlip.status == status.upper())

    slips = (await db.execute(stmt.limit(limit))).scalars().all()

    # Calculate portfolio aggregates across all settled slips
    all_stmt = select(UserBetSlip).where(UserBetSlip.user_id == user.id)
    all_slips = (await db.execute(all_stmt)).scalars().all()

    total_wagered = sum(s.stake_amount for s in all_slips)
    settled_slips = [s for s in all_slips if s.status in ("WON", "LOST")]
    total_pnl = sum(s.pnl for s in settled_slips)
    settled_wagered = sum(s.stake_amount for s in settled_slips)
    roi_pct = round((total_pnl / settled_wagered) * 100.0, 2) if settled_wagered > 0 else 0.0

    won_count = sum(1 for s in settled_slips if s.status == "WON")
    win_rate = round((won_count / len(settled_slips)) * 100.0, 1) if settled_slips else 0.0
    pending_count = sum(1 for s in all_slips if s.status == "PENDING")

    # Average CLV
    clv_values = [s.clv_edge_pct for s in all_slips if s.clv_edge_pct is not None]
    avg_clv = round(sum(clv_values) / len(clv_values), 2) if clv_values else 0.0

    return {
        "portfolio_summary": {
            "total_bets": len(all_slips),
            "pending_bets": pending_count,
            "settled_bets": len(settled_slips),
            "total_wagered": round(total_wagered, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": roi_pct,
            "win_rate_pct": win_rate,
            "avg_clv_edge_pct": avg_clv,
        },
        "slips": [
            {
                "id": str(s.id),
                "prediction_id": str(s.prediction_id) if s.prediction_id else None,
                "match_title": s.match_title,
                "league_name": s.league_name,
                "selection": s.selection,
                "legs": s.legs,
                "bookmaker": s.bookmaker,
                "odds_origin": s.odds_origin,
                "currency": "ZAR",
                "odds_taken": s.odds_taken,
                "stake_amount": s.stake_amount,
                "status": s.status,
                "pnl": round(s.pnl, 2),
                "closing_odds": s.closing_odds,
                "clv_edge_pct": s.clv_edge_pct,
                "notes": s.notes,
                "created_at": s.created_at.isoformat() if s.created_at else "",
                "settled_at": s.settled_at.isoformat() if s.settled_at else None,
            }
            for s in slips
        ],
    }


@router.post("")
async def create_bet_slip(
    payload: CreateBetSlipRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Log a new bet into the user's personal journal."""
    legs = []
    for leg in payload.legs or []:
        data = leg.model_dump(mode="json")
        data["status"] = "PENDING"
        data["odds_origin"] = "manual_unverified"
        if leg.quote_id:
            quote = await db.get(SABookmakerOdds, leg.quote_id)
            fixture = await db.get(Fixture, quote.fixture_id) if quote and quote.fixture_id else None
            if not quote or not fixture or not observed_quote(quote, fixture):
                raise HTTPException(409, "Odds are unavailable, suspended or stale; refresh the bookmaker quote")
            price = getattr(quote, MARKETS.get(leg.selection, ""), None)
            if price != leg.odds_taken or quote.bookmaker != leg.bookmaker:
                raise HTTPException(409, "Odds or bookmaker changed; review the latest price")
            prediction_id = None
            for prediction_ref in (leg.prediction_id, quote.prediction_id):
                if not prediction_ref:
                    continue
                pred = await db.get(Prediction, prediction_ref)
                if pred and pred.league_id == fixture.league_id and pred.home_team == fixture.home_team and pred.away_team == fixture.away_team and pred.match_date == fixture.match_date:
                    prediction_id = str(pred.id)
                    break
            data.update(match_title=quote.match_title, match_date=quote.match_date.isoformat(),
                prediction_id=prediction_id,
                odds_origin="observed", source_url=quote.source_url, quoted_at=quote.scraped_at.isoformat(),
                fixture_id=str(fixture.id))
        elif leg.prediction_id:
            pred = await db.get(Prediction, leg.prediction_id)
            if not pred or leg.match_title != f"{pred.home_team} vs {pred.away_team}":
                raise HTTPException(422, "Prediction does not match this leg")
            data["match_date"] = pred.match_date.isoformat() if pred.match_date else None
        legs.append(data)
    event_keys = [leg.get("fixture_id") or (leg["match_title"].casefold(), leg.get("match_date")) for leg in legs]
    if len(set(event_keys)) != len(event_keys):
        raise HTTPException(422, "Duplicate event in accumulator")
    if payload.prediction_id and not legs:
        pred = await db.get(Prediction, payload.prediction_id)
        if not pred or payload.match_title != f"{pred.home_team} vs {pred.away_team}":
            raise HTTPException(422, "Prediction does not match this bet")
    odds_taken = math.prod(leg["odds_taken"] for leg in legs) if legs else payload.odds_taken
    stake = money(payload.stake_amount)
    clv_edge = None
    if payload.closing_odds and payload.closing_odds > 1.0:
        clv_edge = round(((odds_taken / payload.closing_odds) - 1.0) * 100.0, 2)

    slip = UserBetSlip(
        user_id=user.id,
        prediction_id=payload.prediction_id,
        match_title=" + ".join(leg["match_title"] for leg in legs)[:250] if legs else payload.match_title,
        league_name=payload.league_name or "Top League",
        selection=("ACCUMULATOR" if len(legs) > 1 else legs[0]["selection"]) if legs else payload.selection,
        odds_taken=odds_taken,
        stake_amount=stake,
        legs=legs or None,
        bookmaker=legs[0]["bookmaker"] if legs else payload.bookmaker,
        odds_origin="observed" if legs and all(leg["odds_origin"] == "observed" for leg in legs) else "manual_unverified",
        status="PENDING",
        pnl=0.0,
        closing_odds=payload.closing_odds,
        clv_edge_pct=clv_edge,
        notes=payload.notes,
    )
    db.add(slip)
    await db.commit()
    await db.refresh(slip)

    return {
        "status": "success",
        "slip_id": str(slip.id),
        "message": f"Bet logged for {slip.match_title} ({slip.selection} @{slip.odds_taken})",
    }


@router.patch("/{slip_id}")
async def update_bet_slip(
    slip_id: uuid.UUID,
    payload: UpdateBetSlipRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update status, settlement, or notes for a bet slip."""
    stmt = select(UserBetSlip).where(UserBetSlip.id == slip_id, UserBetSlip.user_id == user.id)
    slip = (await db.execute(stmt)).scalar_one_or_none()
    if not slip:
        raise HTTPException(status_code=404, detail="Bet slip not found")

    if payload.status:
        st = payload.status.upper()
        slip.status = st
        if st == "PENDING":
            slip.pnl = 0.0
            slip.settled_at = None
        if st in ("WON", "LOST", "VOID"):
            slip.settled_at = datetime.now(timezone.utc)
            if st == "WON":
                effective_odds = math.prod(leg["odds_taken"] for leg in slip.legs if leg.get("status") != "VOID") if slip.legs else slip.odds_taken
                slip.pnl = money(slip.stake_amount * (effective_odds - 1.0))
            elif st == "LOST":
                slip.pnl = round(-slip.stake_amount, 2)
            else:
                slip.pnl = 0.0

    if payload.closing_odds is not None:
        slip.closing_odds = payload.closing_odds
        if payload.closing_odds > 1.0:
            slip.clv_edge_pct = round(((slip.odds_taken / payload.closing_odds) - 1.0) * 100.0, 2)

    if payload.notes is not None:
        slip.notes = payload.notes

    await db.commit()
    await db.refresh(slip)

    return {"status": "success", "slip_id": str(slip.id), "new_status": slip.status, "pnl": slip.pnl}


@router.delete("/{slip_id}")
async def delete_bet_slip(
    slip_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a bet slip from the journal."""
    stmt = select(UserBetSlip).where(UserBetSlip.id == slip_id, UserBetSlip.user_id == user.id)
    slip = (await db.execute(stmt)).scalar_one_or_none()
    if not slip:
        raise HTTPException(status_code=404, detail="Bet slip not found")

    await db.delete(slip)
    await db.commit()
    return {"status": "success", "deleted_slip_id": str(slip_id)}


@router.post("/auto-settle")
async def auto_settle_user_slips(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Auto-settle pending bet slips linked to settled match predictions.
    """
    slips = (await db.execute(select(UserBetSlip).where(
        UserBetSlip.user_id == user.id, UserBetSlip.status == "PENDING").with_for_update())).scalars().all()
    ids = set()
    for slip in slips:
        for leg in slip.legs or [{"prediction_id": str(slip.prediction_id) if slip.prediction_id else None}]:
            if leg.get("prediction_id"):
                ids.add(uuid.UUID(leg["prediction_id"]))
    now = datetime.now(timezone.utc)
    predictions = (await db.execute(select(Prediction).where(Prediction.id.in_(ids),
        verified_result(), Prediction.match_date < now))).scalars().all() if ids else []
    by_id = {str(pred.id): pred for pred in predictions}
    settled_count = 0
    for slip in slips:
        legs = [dict(leg) for leg in (slip.legs or [{"prediction_id": str(slip.prediction_id),
            "selection": slip.selection, "odds_taken": slip.odds_taken, "status": "PENDING"}])]
        for leg in legs:
            pred = by_id.get(leg.get("prediction_id"))
            if pred:
                from backend.app.services.prediction_markets import settle_prediction_selection
                result = settle_prediction_selection(leg["selection"], pred)
                if result:
                    leg["status"] = result
        state, pnl = settle_legs(legs, slip.stake_amount)
        if slip.legs:
            slip.legs = legs
        if state != "PENDING":
            slip.status, slip.pnl, slip.settled_at = state, pnl, now
            settled_count += 1
    await db.commit()

    return {"status": "success", "settled_slips_count": settled_count}
