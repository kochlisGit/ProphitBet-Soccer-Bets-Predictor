from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import require_plan
from backend.app.db.models import Fixture, League, User
from backend.app.db.session import get_db

router = APIRouter()


class FixtureOut(BaseModel):
    id: str
    league_id: int
    league_name: str
    country: Optional[str] = None
    home_team: str
    away_team: str
    match_date: Optional[str] = None
    odds_1: Optional[float] = None
    odds_x: Optional[float] = None
    odds_2: Optional[float] = None
    source_url: Optional[str] = None
    fetched_at: Optional[str] = None
    predicted: bool

    class Config:
        from_attributes = True


@router.get("/upcoming", response_model=list[FixtureOut])
async def get_upcoming_fixtures(
    league_id: Optional[int] = Query(None, description="Filter by exact league ID"),
    league: Optional[str] = Query(None, description="Filter by league name"),
    limit: int = Query(500, ge=10, le=1000),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_plan("pro")),
):
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)

    stmt = (
        select(Fixture, League.name.label("league_name"), League.country.label("country"))
        .join(League, Fixture.league_id == League.id)
        .where(
            Fixture.match_date >= now,
            Fixture.is_current.is_(True),
            Fixture.source_url.isnot(None),
            Fixture.fetched_at >= now - timedelta(hours=48),
        )
    )

    if league_id is not None:
        stmt = stmt.where(Fixture.league_id == league_id)
    elif league and league.lower() != "all":
        norm = league.lower().replace("-", " ").strip()
        norm_words = norm.split()
        if "psl" in norm_words or "betway" in norm_words or "premiership" in norm_words:
            stmt = stmt.where(League.id == 34)
        elif ("premier" in norm_words or "epl" in norm_words) and "russia" not in norm:
            stmt = stmt.where(League.id == 6)
        elif "championship" in norm:
            stmt = stmt.where(League.id == 7)
        elif "la liga" in norm or "laliga" in norm or "spain" in norm:
            stmt = stmt.where(League.id == 28)
        elif ("serie a" in norm or "seriea" in norm) and "brazil" not in norm:
            stmt = stmt.where(League.id == 17)
        elif "bundesliga" in norm and "2" not in norm:
            stmt = stmt.where(League.id == 13)
        elif "ligue 1" in norm or "ligue1" in norm:
            stmt = stmt.where(League.id == 11)
        else:
            stmt = stmt.where(League.name.ilike(f"%{league}%"))

    stmt = stmt.order_by(Fixture.match_date.asc()).limit(limit)
    result = await db.execute(stmt)
    rows = result.all()

    return [
        FixtureOut(
            id=str(f.id),
            league_id=f.league_id,
            league_name=ln,
            country=country,
            home_team=f.home_team,
            away_team=f.away_team,
            match_date=f.match_date.isoformat() if f.match_date else None,
            odds_1=f.odds_1,
            odds_x=f.odds_x,
            odds_2=f.odds_2,
            source_url=f.source_url,
            fetched_at=f.fetched_at.isoformat() if f.fetched_at else None,
            predicted=f.predicted,
        )
        for f, ln, country in rows
    ]


@router.get("/league/{league_id}", response_model=list[FixtureOut])
async def get_league_fixtures(
    league_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_plan("pro")),
):
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    
    league_result = await db.execute(select(League).where(League.id == league_id))
    league = league_result.scalar_one_or_none()
    if league is None:
        raise HTTPException(status_code=404, detail="League not found")
    
    stmt = (
        select(Fixture)
        .where(
            Fixture.league_id == league_id,
            Fixture.match_date >= now,
            Fixture.is_current.is_(True),
            Fixture.source_url.isnot(None),
            Fixture.fetched_at >= now - timedelta(hours=48),
        )
        .order_by(Fixture.match_date.asc())
        .limit(150)
    )
    result = await db.execute(stmt)
    fixtures = result.scalars().all()
    
    fixture_data = [
        {
            "id": str(f.id),
            "league_id": f.league_id,
            "league_name": league.name,
            "country": league.country,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "match_date": f.match_date.isoformat() if f.match_date else None,
            "odds_1": f.odds_1,
            "odds_x": f.odds_x,
            "odds_2": f.odds_2,
            "source_url": f.source_url,
            "fetched_at": f.fetched_at.isoformat() if f.fetched_at else None,
            "predicted": f.predicted,
        }
        for f in fixtures
    ]
    
    return [FixtureOut(**fix) for fix in fixture_data]
