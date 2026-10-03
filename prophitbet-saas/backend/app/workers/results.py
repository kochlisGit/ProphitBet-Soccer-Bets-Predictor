import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from sqlalchemy import or_

from backend.app.config import setup_ml_path
from backend.app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _get_sync_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.config import get_settings
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL_SYNC)
    return sessionmaker(bind=engine)()


@celery_app.task(name="backend.app.workers.results.update_match_results_task", bind=True)
def update_match_results_task(self, all_time: bool = False, lookback_days: Optional[int] = None):
    """
    Match completed fixtures to predictions and fill actual_result / is_correct.

    Compares predictions against the latest league CSV data which contains
    final match scores (Result or FTR column = Full Time Result: H/D/A).
    """
    setup_ml_path()
    session = _get_sync_session()
    try:
        from backend.app.db.models import Prediction, LeagueDataset, League, MiroFishSimulation
        from backend.app.services.league_service import _load_csv_from_s3_sync

        query = (
            session.query(Prediction)
            .filter(
                or_(
                    Prediction.actual_result.is_(None),
                    Prediction.result_source.is_(None),
                    Prediction.result_verified_at.is_(None),
                ),
                Prediction.match_date.isnot(None),
                Prediction.match_date < datetime.now(timezone.utc),
            )
        )

        if not all_time and lookback_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            query = query.filter(Prediction.match_date >= cutoff)

        unresolved = query.all()

        if not unresolved:
            logger.info("No unresolved predictions to update")
            return {"updated": 0, "checked": 0}

        league_ids = set(p.league_id for p in unresolved)

        league_dfs = {}
        live_result_dfs = {}
        live_result_sources = {}
        work_total = len(league_ids) + len(unresolved)
        self.update_state(state="PROGRESS", meta={
            "current": 0, "total": work_total,
            "phase": "Loading verified result sources", "item": None,
        })
        for league_index, lid in enumerate(league_ids, start=1):
            self.update_state(state="PROGRESS", meta={
                "current": league_index - 1, "total": work_total,
                "phase": "Loading verified result sources", "item": f"League {lid}",
            })
            league = session.query(League).filter(League.id == lid).first()
            if league is not None:
                live_df, live_source = _load_fixture_download_results(league)
                if live_df is not None and not live_df.empty:
                    live_result_dfs[lid] = _prepare_result_dates(live_df)
                    live_result_sources[lid] = live_source
            ds = (
                session.query(LeagueDataset)
                .filter(LeagueDataset.league_id == lid)
                .order_by(LeagueDataset.created_at.desc())
                .first()
            )
            if ds is None:
                continue
            try:
                df = _load_csv_from_s3_sync(ds.file_path)
                league_dfs[lid] = _prepare_result_dates(df)
            except Exception as e:
                logger.warning(f"Could not load CSV for league {lid}: {e}")

        self.update_state(state="PROGRESS", meta={
            "current": len(league_ids), "total": work_total,
            "phase": "Checking settled matches", "item": None,
        })

        updated = 0
        for prediction_index, pred in enumerate(unresolved, start=1):
            item = f"{pred.home_team} vs {pred.away_team}"
            self.update_state(state="PROGRESS", meta={
                "current": len(league_ids) + prediction_index - 1,
                "total": work_total, "phase": "Checking settled matches", "item": item,
            })
            match_info = None
            source = None
            current_df = live_result_dfs.get(pred.league_id)
            if current_df is not None and not current_df.empty:
                match_info = _find_match_result(current_df, pred.home_team, pred.away_team, pred.match_date, pred.market_type)
                if match_info is not None:
                    source = live_result_sources[pred.league_id]
            if match_info is None:
                df = league_dfs.get(pred.league_id)
                if df is not None and not df.empty:
                    match_info = _find_match_result(df, pred.home_team, pred.away_team, pred.match_date, pred.market_type)
                    if match_info is not None:
                        source = f"league_dataset:{pred.league_id}:{match_info['date']}"
            if match_info is None:
                continue

            actual = match_info["result"]
            pred.actual_result = actual
            pred.actual_score = match_info["score"]
            pred.result_source = source
            pred.result_verified_at = datetime.now(timezone.utc)
            pred.is_correct = (pred.predicted_result == actual)
            updated += 1

        self.update_state(state="PROGRESS", meta={
            "current": work_total, "total": work_total,
            "phase": "Saving verified results", "item": None,
        })

        session.commit()
        logger.info(f"Updated {updated} prediction results out of {len(unresolved)} unresolved")
        return {"updated": updated, "checked": len(unresolved)}

    finally:
        session.close()


def _load_fixture_download_results(league):
    """Fetch current completed scores from a public JSON season feed."""
    import httpx
    import pandas as pd
    from backend.app.workers.fixtures import FIXTURE_DOWNLOAD_LEAGUES

    slug = FIXTURE_DOWNLOAD_LEAGUES.get((league.country, league.name))
    if not slug:
        return None, None
    now = datetime.now(timezone.utc)
    season = now.year if now.month >= 7 or slug == "mls" else now.year - 1
    url = f"https://fixturedownload.com/feed/json/{slug}-{season}"
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": "ProphitBet/1.0"})
        response.raise_for_status()
        if slug == "champions-league":
            from src.network.leagues.downloaders.public_json import parse_completed_matches
            return pd.DataFrame(parse_completed_matches(response.json(), season, url, now)), url
        rows = []
        for match in response.json():
            home_goals = match.get("HomeTeamScore")
            away_goals = match.get("AwayTeamScore")
            if home_goals is None or away_goals is None:
                continue
            home_goals, away_goals = int(home_goals), int(away_goals)
            rows.append({
                "Date": match.get("DateUtc"),
                "HomeTeam": match.get("HomeTeam"),
                "AwayTeam": match.get("AwayTeam"),
                "FTHG": home_goals,
                "FTAG": away_goals,
                "FTR": "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D"),
            })
        return pd.DataFrame(rows), url
    except Exception as exc:
        logger.warning("Fixture Download results failed for %s - %s: %s", league.country, league.name, exc)
        return None, None


def _prepare_result_dates(df):
    """Parse a dataset's dates once instead of once per prediction."""
    import pandas as pd

    prepared = df.copy()
    if "Date" in prepared.columns:
        raw_dates = prepared["Date"].astype("string").str.strip()
        # Public JSON feeds use unambiguous ISO year-first dates while the
        # historical CSVs commonly use day-first dates. Applying dayfirst=True
        # to both silently turned 2026-09-12 into 9 December 2026.
        iso_mask = raw_dates.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", na=False)
        parsed = pd.Series(pd.NaT, index=prepared.index, dtype="datetime64[ns, UTC]")
        parsed.loc[iso_mask] = pd.to_datetime(
            raw_dates.loc[iso_mask], format="mixed", dayfirst=False, errors="coerce", utc=True
        )
        parsed.loc[~iso_mask] = pd.to_datetime(
            raw_dates.loc[~iso_mask], format="mixed", dayfirst=True, errors="coerce", utc=True
        )
        prepared["_verified_match_date"] = parsed
    return prepared


def _find_match_result(df, home_team: str, away_team: str, match_date=None, market_type="result") -> Optional[Dict[str, Any]]:
    """
    Look up match result in the league DataFrame.
    Supports both 'Result' (ProphitBet format) and 'FTR' (raw football-data format).
    """
    if df is None or df.empty:
        return None
    if "ScoreScope" in df.columns:
        df = df[df["ScoreScope"] == "regulation"]
        if df.empty:
            return None

    import difflib
    import pandas as pd
    from backend.app.services.team_mapping import normalize_team_name

    res_col = "Result" if "Result" in df.columns else ("FTR" if "FTR" in df.columns else None)
    if res_col is None:
        return None

    home_col = "Home" if "Home" in df.columns else ("HomeTeam" if "HomeTeam" in df.columns else None)
    away_col = "Away" if "Away" in df.columns else ("AwayTeam" if "AwayTeam" in df.columns else None)
    hg_col = "HG" if "HG" in df.columns else ("FTHG" if "FTHG" in df.columns else None)
    ag_col = "AG" if "AG" in df.columns else ("FTAG" if "FTAG" in df.columns else None)

    if not home_col or not away_col:
        return None

    uniq_teams = list(set(df[home_col].dropna().unique()).union(set(df[away_col].dropna().unique())))
    norm_home = normalize_team_name(home_team, uniq_teams) or home_team
    norm_away = normalize_team_name(away_team, uniq_teams) or away_team

    mask = (
        ((df[home_col].astype(str).str.lower() == norm_home.lower()) | (df[home_col].astype(str).str.lower() == home_team.lower())) &
        ((df[away_col].astype(str).str.lower() == norm_away.lower()) | (df[away_col].astype(str).str.lower() == away_team.lower()))
    )

    matches = df[mask]
    if matches.empty:
        close_h = difflib.get_close_matches(home_team, uniq_teams, n=1, cutoff=0.65)
        close_a = difflib.get_close_matches(away_team, uniq_teams, n=1, cutoff=0.65)
        if close_h and close_a:
            fuzzy_mask = (df[home_col] == close_h[0]) & (df[away_col] == close_a[0])
            matches = df[fuzzy_mask]

    if matches.empty:
        return None

    # If multiple matches occurred between these teams, match closest by date
    if match_date is None or "Date" not in df.columns:
        return None
    try:
        target_date = match_date.date() if hasattr(match_date, "date") else match_date
        df_dates = (
            matches["_verified_match_date"]
            if "_verified_match_date" in matches.columns
            else _prepare_result_dates(matches)["_verified_match_date"]
        )
        date_matches = matches[df_dates.dt.date == target_date]
        if date_matches.empty:
            return None
        matches = date_matches
    except Exception:
        return None

    row = matches.iloc[-1]
    res_val = str(row[res_col]).strip().upper()
    if res_val not in ("H", "D", "A"):
        return None

    if not hg_col or not ag_col or pd.isna(row.get(hg_col)) or pd.isna(row.get(ag_col)):
        return None
    home_goals, away_goals = int(row[hg_col]), int(row[ag_col])
    score_result = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")
    if score_result != res_val:
        logger.warning("Rejected inconsistent result row: score=%s-%s result=%s", home_goals, away_goals, res_val)
        return None
    score_str = f"{home_goals}-{away_goals}"
    from src.preprocessing.utils.target import construct_targets, target_labels, parse_target
    try:
        target = parse_target(market_type or "result")
        observed = dict(row)
        observed.update(Result=res_val, HG=row[hg_col], AG=row[ag_col])
        actual = target_labels(target)[int(construct_targets(pd.DataFrame([observed]), target)[0])]
    except (ValueError, KeyError):
        return None

    return {
        "result": actual,
        "score": score_str,
        "date": str(row["Date"]),
    }
