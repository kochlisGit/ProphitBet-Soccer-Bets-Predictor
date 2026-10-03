"""
Seed script for ProphitBet SaaS.

Run inside Docker:
    docker-compose exec backend python -m backend.seed

Or locally (with DB running):
    python -m backend.seed
"""
import json
import sys
from pathlib import Path

# Ensure the app root is on sys.path
app_root = Path(__file__).resolve().parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from backend.app.config import get_settings, get_storage_path

settings = get_settings()


def get_sync_engine():
    from sqlalchemy import create_engine
    return create_engine(settings.DATABASE_URL_SYNC, echo=False)


def create_tables(engine):
    from backend.app.db.session import Base
    from backend.app.db import models  # noqa: F401 — register all models
    Base.metadata.create_all(engine)
    print("[OK] Database tables created/verified")


def seed_admin(session):
    from backend.app.db.models import User, Subscription
    from backend.app.auth.jwt import hash_password

    email = "admin@prophitbet.com"
    existing = session.query(User).filter(User.email == email).first()
    if existing:
        print(f"[SKIP] Admin user already exists: {email}")
        return existing

    user = User(
        email=email,
        password_hash=hash_password("admin123!"),
        name="Admin",
        provider="email",
        is_active=True,
        is_admin=True,
    )
    session.add(user)
    session.flush()

    sub = Subscription(user_id=user.id, plan="elite", status="active")
    session.add(sub)
    session.flush()

    print(f"[OK] Admin user created: {email} / admin123!")
    return user


def seed_leagues(session):
    from backend.app.db.models import League

    storage_path = get_storage_path()
    catalog_path = storage_path / "network" / "leagues.json"

    if not catalog_path.exists():
        print(f"[ERROR] League catalog not found at {catalog_path}")
        return 0

    with open(catalog_path) as f:
        catalog = json.load(f)

    def _norm_fixture(u: str) -> str:
        s = (u or "").strip()
        if s.startswith("ttps://"):
            return "h" + s
        return s

    created = 0
    for entry in catalog.get("leagues", []):
        existing = (
            session.query(League)
            .filter(League.country == entry["country"], League.name == entry["name"])
            .first()
        )
        if existing:
            continue

        lg = League(
            country=entry["country"],
            name=entry["name"],
            category=entry.get("category", "main"),
            url=entry["url"],
            fixture_url=_norm_fixture(entry.get("fixture", "")),
            start_year=entry.get("start_year", 2005),
            is_active=True,
        )
        session.add(lg)
        created += 1

    session.flush()
    print(f"[OK] Seeded {created} leagues from catalog ({catalog_path.name})")
    return created


def main():
    print("=== ProphitBet SaaS Seed ===")
    print(f"DB: {settings.DATABASE_URL_SYNC}")

    engine = get_sync_engine()
    create_tables(engine)

    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        seed_admin(session)
        seed_leagues(session)
        session.commit()
        print("=== Seed complete ===")
    except Exception as e:
        session.rollback()
        print(f"[ERROR] Seed failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
