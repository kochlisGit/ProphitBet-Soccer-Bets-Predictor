import asyncio
import logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from backend.app.config import get_settings, setup_ml_path
from backend.app.workers.celery_app import celery_app
from backend.app.workers.sync_utils import assert_enough_disk_for_sync

logger = logging.getLogger(__name__)

# Thread pool for blocking operations (downloads, stats computation)
_thread_pool: Optional[ThreadPoolExecutor] = None


def _download_public_results_fallback(lg):
    """Return factual score history for leagues unsupported by football-data CSVs."""
    if (lg.country, lg.name) != ("South Africa", "Betway-Premiership"):
        return None

    import httpx
    import pandas as pd

    now = datetime.now(timezone.utc)
    start_year = now.year if now.month >= 7 else now.year - 1
    season_label = f"{start_year}-{start_year + 1}"
    url = (
        "https://www.thesportsdb.com/api/v1/json/123/eventsseason.php"
        f"?id=4802&s={season_label}"
    )
    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    rows = []
    for event in response.json().get("events") or []:
        home = event.get("strHomeTeam")
        away = event.get("strAwayTeam")
        date = event.get("dateEvent")
        hg = event.get("intHomeScore")
        ag = event.get("intAwayScore")
        if not (home and away and date) or hg is None or ag is None:
            continue
        hg, ag = int(hg), int(ag)
        rows.append({
            "Date": date,
            "Season": start_year,
            "Home": home,
            "Away": away,
            "HG": hg,
            "AG": ag,
            "Result": "H" if hg > ag else ("A" if ag > hg else "D"),
            "1": pd.NA,
            "X": pd.NA,
            "2": pd.NA,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["Date", "Home"]).drop_duplicates()
    result_pos = df.columns.get_loc("Result") + 1
    df.insert(result_pos, "Result-U/O", (df["HG"] + df["AG"]).ge(2.5).map({True: "O", False: "U"}))
    df["Week"] = df.groupby(["Season", "Home"]).cumcount() + 1
    logger.info("Loaded %s factual results for %s from %s", len(df), lg.name, url)
    return df


def _get_thread_pool() -> ThreadPoolExecutor:
    """Get or create thread pool for blocking operations."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=4)
    return _thread_pool


def _get_sync_session():
    """Create a synchronous DB session for Celery tasks."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.config import get_settings
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL_SYNC)
    return sessionmaker(bind=engine)()


def _sync_one_league(session, lg, main_dl, extra_dl, stats_engine) -> bool:
    """Download, compute stats, upload CSV for one DB league row. Commits on success."""
    from backend.app.db.models import LeagueDataset
    from src.network.leagues.league import League as CoreLeague

    core_league = CoreLeague(
        country=lg.country,
        name=lg.name,
        start_year=lg.start_year,
        category=lg.category,
        url=lg.url,
        fixture=lg.fixture_url or "",
    )

    downloader = main_dl if lg.category == "main" else extra_dl
    if lg.category == "public-json":
        from src.network.leagues.downloaders.public_json import PublicJsonLeagueDownloader
        downloader = PublicJsonLeagueDownloader()
    
    # Celery tasks are already synchronous worker processes. Running futures on
    # a second event loop caused every download to escape in a different loop.
    try:
        df = downloader.download(core_league, lg.start_year)
    except Exception:
        df = _download_public_results_fallback(lg)
        if df is None:
            raise
    
    if df is None or df.empty:
        logger.warning(f"No data downloaded for {lg.country} - {lg.name}")
        return False

    # Reject bad raw results before expensive feature computation.
    from backend.app.services.dataset_validation import validate_dataset
    validate_dataset(df)
    stat_cols = stats_engine.get_basic_stat_columns() + stats_engine.get_extended_stat_columns()
    if lg.category == "public-json":
        # Cup phases have only 3-4 home/away games per team each season.
        # Reuse the shared engine with a shorter, explicit history window.
        from src.preprocessing.statistics import StatisticsEngine
        stats_engine = StatisticsEngine(match_history_window=1, goal_diff_margin=2)
    df = stats_engine.compute_stats(df, stat_cols)

    existing = (session.query(LeagueDataset).filter(LeagueDataset.league_id == lg.id)
                .order_by(LeagueDataset.created_at.desc()).first())
    from backend.app.services.dataset_validation import validate_dataset
    validate_dataset(df, existing.row_count if existing else 0)

    # Immutable objects preserve the previous DB-selected snapshot if commit fails.
    from uuid import uuid4
    s3_key = f"leagues/{lg.id}/snapshots/{uuid4().hex}.csv"
    loop = asyncio.new_event_loop()
    try:
        from backend.app.services.league_service import upload_csv_to_s3
        loop.run_until_complete(upload_csv_to_s3(df, s3_key))
    finally:
        loop.close()

    existing = (
        session.query(LeagueDataset)
        .filter(LeagueDataset.league_id == lg.id)
        .order_by(LeagueDataset.created_at.desc())
        .first()
    )
    if existing:
        existing.file_path = s3_key
        existing.row_count = len(df)
    else:
        ds = LeagueDataset(league_id=lg.id, file_path=s3_key, row_count=len(df))
        session.add(ds)

    lg.last_synced_at = datetime.now(timezone.utc)
    session.commit()
    logger.info(f"Synced {lg.country} - {lg.name}: {len(df)} rows")
    return True


def sync_single_league_by_id(league_id: int) -> dict:
    """
    Refresh one league's dataset from football-data and upload to S3.
    Used by result-ingestion to pull latest scores before matching predictions.
    """
    setup_ml_path()
    from src.network.leagues.downloaders.main import MainLeagueDownloader
    from src.network.leagues.downloaders.extra import ExtraLeagueDownloader
    from src.preprocessing.statistics import StatisticsEngine
    from backend.app.db.models import League as LeagueModel

    session = _get_sync_session()
    try:
        lg = session.query(LeagueModel).filter(LeagueModel.id == league_id).first()
        if lg is None:
            return {"ok": False, "error": "League not found"}
        main_dl = MainLeagueDownloader()
        extra_dl = ExtraLeagueDownloader()
        stats_engine = StatisticsEngine(match_history_window=10, goal_diff_margin=2)
        try:
            ok = _sync_one_league(session, lg, main_dl, extra_dl, stats_engine)
            return {"ok": ok, "league_id": league_id}
        except Exception as e:
            logger.error(f"sync_single_league_by_id({league_id}): {e}")
            session.rollback()
            return {"ok": False, "error": str(e)}
    finally:
        session.close()


@celery_app.task(name="backend.app.workers.data_sync.sync_all_leagues_task", bind=True)
def sync_all_leagues_task(
    self,
    force_resync: bool = False,
    skip_if_synced_within_hours: int | None = None,
):
    """Download/update match data for all active leagues and upload to S3."""
    setup_ml_path()
    settings = get_settings()
    hours = (
        skip_if_synced_within_hours
        if skip_if_synced_within_hours is not None
        else settings.SYNC_SKIP_IF_NEWER_THAN_HOURS
    )

    if settings.SYNC_MIN_FREE_DISK_MB > 0:
        ok_disk, free_mb = assert_enough_disk_for_sync(
            settings.SYNC_MIN_FREE_DISK_MB,
            settings.DISK_CHECK_PATH,
        )
        if not ok_disk:
            return {
                "error": "low_disk",
                "free_mb": round(free_mb, 1),
                "required_mb": settings.SYNC_MIN_FREE_DISK_MB,
                "path": settings.DISK_CHECK_PATH,
            }

    from src.network.leagues.downloaders.main import MainLeagueDownloader
    from src.network.leagues.downloaders.extra import ExtraLeagueDownloader
    from src.preprocessing.statistics import StatisticsEngine

    session = _get_sync_session()
    try:
        from backend.app.db.models import League as LeagueModel
        from backend.app.db.models import LeagueDataset

        leagues = session.query(LeagueModel).filter(LeagueModel.is_active == True).all()
        logger.info(f"Starting data sync for {len(leagues)} leagues (force_resync={force_resync}, hours={hours})")

        main_dl = MainLeagueDownloader()
        extra_dl = ExtraLeagueDownloader()
        stats_engine = StatisticsEngine(match_history_window=10, goal_diff_margin=2)

        now = datetime.now(timezone.utc)
        synced = 0
        skipped = 0
        failed = 0
        
        # Filter leagues that need syncing
        leagues_to_sync = []
        for lg in leagues:
            if (
                not force_resync
                and hours > 0
                and lg.last_synced_at is not None
            ):
                ds = (
                    session.query(LeagueDataset)
                    .filter(LeagueDataset.league_id == lg.id)
                    .first()
                )
                if ds is not None and (now - lg.last_synced_at) < timedelta(hours=hours):
                    logger.info(
                        "Skipping %s - %s (synced recently, dataset exists)",
                        lg.country,
                        lg.name,
                    )
                    skipped += 1
                    continue
            leagues_to_sync.append(lg)
        
        logger.info(f"Processing {len(leagues_to_sync)} leagues (skipped {skipped})")
        self.update_state(state="PROGRESS", meta={
            "current": skipped, "total": len(leagues),
            "phase": "Preparing league data", "item": None,
        })
        
        # Process leagues sequentially but with async thread pool for blocking ops
        for index, lg in enumerate(leagues_to_sync, start=1):
            item = f"{lg.country} — {lg.name}"
            self.update_state(state="PROGRESS", meta={
                "current": skipped + index - 1, "total": len(leagues),
                "phase": "Downloading and processing league", "item": item,
            })
            try:
                if _sync_one_league(session, lg, main_dl, extra_dl, stats_engine):
                    synced += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.error(f"Failed to sync {lg.country} - {lg.name}: {e}")
                session.rollback()
                continue
            finally:
                self.update_state(state="PROGRESS", meta={
                    "current": skipped + index, "total": len(leagues),
                    "phase": "League processed", "item": item,
                })

        if failed and not synced and leagues_to_sync:
            raise RuntimeError(f'All {failed} attempted league syncs failed; previous datasets retained')
        return {"status": "partial" if failed else "complete", "synced": synced,
                "skipped": skipped, "failed": failed, "total": len(leagues)}
    finally:
        session.close()
        # Invalidate league cache after sync
        try:
            from backend.app.services.cache import get_cache
            cache = get_cache()
            cache.delete_pattern("league:list")
            cache.delete_pattern("league:metadata:*")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache after sync: {e}")


@celery_app.task(name="backend.app.workers.data_sync.seed_leagues_from_catalog")
def seed_leagues_from_catalog():
    """Seed the leagues table from the desktop app's leagues.json catalog."""
    import json
    from pathlib import Path

    setup_ml_path()
    from backend.app.config import get_storage_path
    catalog_path = get_storage_path() / "network" / "leagues.json"

    if not catalog_path.exists():
        logger.error(f"League catalog not found at {catalog_path}")
        return {"error": "Catalog not found"}

    with open(catalog_path) as f:
        catalog = json.load(f)

    session = _get_sync_session()
    try:
        from backend.app.db.models import League as LeagueModel
        def _norm_fixture(u: str) -> str:
            s = (u or "").strip()
            if s.startswith("ttps://"):
                return "h" + s
            return s

        created = 0
        for entry in catalog.get("leagues", []):
            existing = (
                session.query(LeagueModel)
                .filter(LeagueModel.country == entry["country"], LeagueModel.name == entry["name"])
                .first()
            )
            if existing:
                continue
            lg = LeagueModel(
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

        session.commit()
        logger.info(f"Seeded {created} leagues from catalog")
        return {"created": created}
    finally:
        session.close()
