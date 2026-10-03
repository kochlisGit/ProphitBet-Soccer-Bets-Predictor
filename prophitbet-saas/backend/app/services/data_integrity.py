"""Shared SQL predicates for publishing sourced match data."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select, func
from backend.app.db.models import Fixture, Prediction


def verified_result(now=None):
    now = now or datetime.now(timezone.utc)
    return and_(
        Prediction.actual_result.isnot(None),
        Prediction.actual_score.isnot(None),
        Prediction.actual_score.regexp_match(r'^\s*[0-9]+\s*-\s*[0-9]+\s*$'),
        Prediction.result_source.isnot(None),
        Prediction.result_source != '',
        func.length(func.trim(Prediction.result_source)) > 0,
        Prediction.match_date < now,
        Prediction.result_verified_at.isnot(None),
        Prediction.result_verified_at <= now,
    )


def publishable_prediction(now=None):
    now = now or datetime.now(timezone.utc)
    current_fixture = select(Fixture.id).where(
        Fixture.league_id == Prediction.league_id,
        Fixture.home_team == Prediction.home_team,
        Fixture.away_team == Prediction.away_team,
        Fixture.match_date == Prediction.match_date,
        Fixture.is_current.is_(True),
        Fixture.source_url.isnot(None),
        Fixture.fetched_at >= now - timedelta(hours=48),
        Fixture.fetched_at <= now,
    ).correlate(Prediction).exists()
    return or_(
        and_(Prediction.match_date < now, verified_result(now)),
        and_(Prediction.match_date >= now, Prediction.actual_result.is_(None), current_fixture),
    )
