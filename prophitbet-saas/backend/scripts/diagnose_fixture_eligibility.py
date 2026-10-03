"""One-off diagnostics: house-model leagues vs fixture dates. Run: python -m backend.scripts.diagnose_fixture_eligibility"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

app_root = Path(__file__).resolve().parent.parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from backend.app.config import get_settings


def main():
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL_SYNC)
    Session = sessionmaker(bind=engine)
    session = Session()
    now = datetime.now(timezone.utc)

    from backend.app.db.models import Fixture, League, TrainedModel

    league_ids = (
        session.query(TrainedModel.league_id)
        .filter(TrainedModel.is_house_model == True)
        .distinct()
        .all()
    )
    rows = []
    for (lid,) in league_ids:
        lg = session.query(League).filter(League.id == lid).first()
        if lg:
            rows.append((lid, lg.country, lg.name))

    print(f"now (UTC) = {now.isoformat()}")
    print(f"House-model leagues: {len(rows)}\n")

    for league_id, country, name in rows:
        total = session.query(func.count()).select_from(Fixture).filter(Fixture.league_id == league_id).scalar()
        not_pred = (
            session.query(func.count())
            .select_from(Fixture)
            .filter(Fixture.league_id == league_id, Fixture.predicted == False)
            .scalar()
        )
        future_or_null = (
            session.query(func.count())
            .select_from(Fixture)
            .filter(
                Fixture.league_id == league_id,
                Fixture.predicted == False,
                (Fixture.match_date >= now) | (Fixture.match_date.is_(None)),
            )
            .scalar()
        )
        mx = session.query(func.max(Fixture.match_date)).filter(Fixture.league_id == league_id).scalar()
        mn = session.query(func.min(Fixture.match_date)).filter(Fixture.league_id == league_id).scalar()
        null_dates = (
            session.query(func.count())
            .select_from(Fixture)
            .filter(Fixture.league_id == league_id, Fixture.match_date.is_(None))
            .scalar()
        )
        print(f"id={league_id} {country} - {name}")
        print(f"  fixtures total={total}, not_predicted={not_pred}, eligible(future|null)={future_or_null}")
        print(f"  match_date min={mn} max={mx} null_dates={null_dates}")

    session.close()


if __name__ == "__main__":
    main()
