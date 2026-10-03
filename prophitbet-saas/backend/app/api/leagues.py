from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import get_optional_user
from backend.app.db.models import League, LeagueDataset, User
from backend.app.db.session import get_db
from backend.app.services.cache import get_cached_league_list, cache_league_list, get_cached_league_metadata, cache_league_metadata

router = APIRouter()


class LeagueOut(BaseModel):
    id: int
    country: str
    name: str
    category: str
    start_year: int
    is_active: bool
    last_synced_at: Optional[str] = None

    class Config:
        from_attributes = True


class LeagueDetailOut(LeagueOut):
    url: str
    fixture_url: Optional[str] = None
    dataset_count: int = 0
    total_rows: int = 0


@router.get("", response_model=list[LeagueOut])
async def list_leagues(
    country: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    _user: Optional[User] = Depends(get_optional_user),
):
    # Try cache first (only for unfiltered requests)
    if not country and not category:
        cached = get_cached_league_list()
        if cached is not None:
            return [LeagueOut(**lg) for lg in cached]
    
    stmt = select(League).where(League.is_active == True)
    if country:
        stmt = stmt.where(League.country == country)
    if category:
        stmt = stmt.where(League.category == category)
    stmt = stmt.order_by(League.country, League.name)
    result = await db.execute(stmt)
    leagues = result.scalars().all()
    league_data = [
        {
            "id": lg.id,
            "country": lg.country,
            "name": lg.name,
            "category": lg.category,
            "start_year": lg.start_year,
            "is_active": lg.is_active,
            "last_synced_at": lg.last_synced_at.isoformat() if lg.last_synced_at else None,
        }
        for lg in leagues
    ]
    
    # Cache unfiltered results
    if not country and not category:
        cache_league_list(league_data)
    
    return [LeagueOut(**lg) for lg in league_data]


@router.get("/{league_id}", response_model=LeagueDetailOut)
async def get_league(league_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(League).where(League.id == league_id))
    league = result.scalar_one_or_none()
    if league is None:
        raise HTTPException(status_code=404, detail="League not found")

    ds_stats = await db.execute(
        select(func.count(), func.coalesce(func.sum(LeagueDataset.row_count), 0))
        .where(LeagueDataset.league_id == league_id)
    )
    ds_count, total_rows = ds_stats.one()

    return LeagueDetailOut(
        id=league.id,
        country=league.country,
        name=league.name,
        category=league.category,
        start_year=league.start_year,
        is_active=league.is_active,
        last_synced_at=league.last_synced_at.isoformat() if league.last_synced_at else None,
        url=league.url,
        fixture_url=league.fixture_url,
        dataset_count=ds_count,
        total_rows=total_rows,
    )


@router.get("/{league_id}/stats")
async def get_league_stats(league_id: int, db: AsyncSession = Depends(get_db)):
    """Return descriptive statistics for the league's latest dataset."""
    import math
    from backend.app.services.league_service import get_league_dataframe
    df = await get_league_dataframe(league_id, db)
    if df is None:
        raise HTTPException(status_code=404, detail="No data available for this league")

    numeric = df.select_dtypes(include="number")
    stats = numeric.describe().round(3).to_dict()
    
    # Sanitize stats dict for JSON compliance (replace NaN/Inf with None)
    def sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj
    
    sanitized_stats = sanitize(stats)
    
    return {"league_id": league_id, "statistics": sanitized_stats, "rows": len(df), "columns": list(df.columns)}


@router.get("/{league_id}/table")
async def get_league_table(
    league_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=10, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Return paginated league data as JSON records."""
    import math
    from backend.app.services.league_service import get_league_dataframe
    df = await get_league_dataframe(league_id, db)
    if df is None:
        raise HTTPException(status_code=404, detail="No data available for this league")

    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end]

    # Convert DataFrame to records, replacing NaN/Inf with None for JSON compliance
    records = page_df.to_dict(orient="records")
    def sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj
    
    sanitized_records = [sanitize(record) for record in records]

    return {
        "league_id": league_id,
        "page": page,
        "per_page": per_page,
        "total": total,
        "columns": list(df.columns),
        "data": sanitized_records,
    }
