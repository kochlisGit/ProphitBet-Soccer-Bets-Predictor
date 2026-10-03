"""FastAPI Routes for South African Betting Markets & Perfect Bet Slip Engine.

Exposes:
- Paired Odds Comparison Matrix (Hollywoodbets, Betway, Sportingbet, Supabets)
- The Perfect Bet Slip Engine (Banker, Value Acca, High-Yield Super Slip, Guaranteed SureBet)
- Cross-Bookmaker Arbitrage (SureBets) with ZAR Bankroll Staking Calculator
- Positive Expected Value (+EV) Bets on SA Bookmakers vs ProphitBet AI
- Bookmaker Vigorish / Margin Comparative Analysis
- One-Click Save of Perfect Slips to User Bet Journal
- On-Demand Scrape and Odds Refresh
"""

from typing import Any, Dict, List, Optional, Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import get_current_user, require_admin
from backend.app.db.models import User, UserBetSlip
from backend.app.db.session import get_db
from backend.app.services.sa_odds_service import sa_odds_service
from backend.app.api.betslip import CreateBetSlipRequest, create_bet_slip

router = APIRouter()


SavePerfectSlipRequest = CreateBetSlipRequest


@router.get('/data-quality')
async def bookmaker_data_quality(db: AsyncSession = Depends(get_db), user: User = Depends(require_admin)):
    from backend.app.services.bookmaker_feed import data_quality_report
    return await data_quality_report(db)


class AISlipRequest(BaseModel):
    bookmaker: Literal['HOLLYWOODBETS', 'BETWAY']
    legs: int = Field(default=2, ge=1, le=6)
    min_probability: float = Field(default=.5, ge=.34, le=.95, allow_inf_nan=False)
    strategy: Literal['confidence', 'value'] = 'confidence'
    min_ev: float = Field(default=3, ge=0, le=50, allow_inf_nan=False)
    draft: bool = False
    market_type: str = "result"

    @field_validator("market_type")
    @classmethod
    def valid_market(cls, value):
        from backend.app.services.prediction_markets import parse_target
        return value if value == "all" else parse_target(value).value


@router.post('/ai-slip')
async def build_ai_slip(payload: AISlipRequest, db: AsyncSession = Depends(get_db),
                        user: User = Depends(get_current_user)):
    """Suggest a slip without placing bets, saving entries, or retraining models."""
    if payload.draft and payload.strategy == 'value':
        raise HTTPException(422, 'Value ranking requires observed bookmaker odds')
    from backend.app.services.ai_slip_builder import generate_ai_slip
    return await generate_ai_slip(db, **payload.model_dump())


@router.get("/odds")
async def get_sa_market_odds(
    league: Optional[str] = Query(None, description="Filter by league (e.g. 'Betway Premiership', 'Premier League')"),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve fresh observed Hollywoodbets and Betway South Africa prices."""
    odds = await sa_odds_service.get_paired_odds_comparison(db, league_filter=league, limit=limit)
    return {
        "total": len(odds),
        "currency": "ZAR",
        "currency_symbol": "R",
        "matches": odds,
        "status": "available" if odds else "unavailable",
        "message": "Only observed South African quotes, at most 15 minutes old. Confirm the accepted price with your bookmaker.",
    }


@router.get("/perfect-slips")
async def get_perfect_bet_slips(
    bankroll: float = Query(1000.0, ge=0.01, le=1000000, allow_inf_nan=False, description="Active user bankroll in South African Rand (ZAR)"),
    db: AsyncSession = Depends(get_db),
):
    """Compatibility endpoint: automatic 'perfect' slips are no longer fabricated."""
    return await sa_odds_service.generate_perfect_bet_slips(db, bankroll_zar=bankroll)


@router.post("/slips/save")
async def save_perfect_slip_to_journal(
    payload: SavePerfectSlipRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Save a recommended leg or slip into the user's personal Bet Journal."""
    if not payload.legs or any(not leg.quote_id for leg in payload.legs):
        raise HTTPException(422, "Select observed quote IDs; manually entered odds belong in the personal journal")
    return await create_bet_slip(payload, db, user)


@router.get("/arbitrage")
async def get_sa_arbitrage_opportunities(
    league: Optional[str] = Query(None, description="Optional league filter"),
    bankroll: float = Query(1000.0, ge=0.01, le=1000000, allow_inf_nan=False, description="Target bankroll in South African Rand (ZAR)"),
    db: AsyncSession = Depends(get_db),
):
    """Report theoretical price discrepancies, never guaranteed profits."""
    return await sa_odds_service.detect_arbitrage_opportunities(db, target_bankroll_zar=bankroll, league_filter=league)


@router.get("/value-bets")
async def get_sa_value_bets(
    league: Optional[str] = Query(None, description="Optional league filter"),
    min_ev: float = Query(3.0, ge=0.0, le=50.0, description="Minimum +EV edge percentage"),
    bankroll: float = Query(2000.0, ge=0.01, le=1000000, allow_inf_nan=False, description="Active user bankroll in ZAR"),
    db: AsyncSession = Depends(get_db),
):
    """Surface +EV value betting opportunities comparing SA odds to ProphitBet AI."""
    return await sa_odds_service.detect_value_bets(db, min_ev_pct=min_ev, zar_bankroll=bankroll, league_filter=league)


@router.get("/margins")
async def get_bookmaker_margins(
    db: AsyncSession = Depends(get_db),
):
    """Analyze and compare bookmaker overround (vigorish/juice) across South African sportsbooks."""
    return await sa_odds_service.get_bookmaker_margin_analysis(db)


@router.post("/sync")
async def sync_sa_odds(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Trigger manual scrape and synchronization of South African bookmaker odds."""
    return await sa_odds_service.sync_all_sa_markets(db)
