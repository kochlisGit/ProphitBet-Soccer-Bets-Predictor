"""Public season archives with source attribution; no fabricated odds or scores."""
from datetime import datetime, timezone
import time

import httpx
import pandas as pd


def parse_completed_matches(matches, season, source, now=None):
    now = now or datetime.now(timezone.utc)
    rows = []
    for match in matches:
        try:
            home, away = match["HomeTeam"], match["AwayTeam"]
            date = datetime.fromisoformat(match["DateUtc"].replace("Z", "+00:00"))
            hg, ag = match.get("HomeTeamScore"), match.get("AwayTeamScore")
            if date.tzinfo is None or date >= now or not home or not away or hg is None or ag is None:
                continue
            if isinstance(hg, bool) or isinstance(ag, bool) or int(hg) != float(hg) or int(ag) != float(ag):
                continue
            hg, ag = int(hg), int(ag)
            if min(hg, ag) < 0:
                continue
            # Archive finals do not identify extra-time/penalty components.
            # Only the group/league phase is unambiguously regulation-time.
            phase_rounds = 8 if season >= 2024 else 6
            regulation = 1 <= int(match.get("RoundNumber", 0)) <= phase_rounds
            rows.append(dict(Date=date.astimezone(timezone.utc).isoformat(), Season=season,
                Home=home, Away=away, HG=hg, AG=ag, Result="H" if hg > ag else "A" if ag > hg else "D",
                **{"Result-U/O": "O" if hg + ag > 2.5 else "U"},
                Source=source, ScoreScope="regulation" if regulation else "provider_final"))
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
    return rows


class PublicJsonLeagueDownloader:
    def download(self, league, start_year):
        if (league.country, league.name) != ("Europe", "Champions-League"):
            raise ValueError("Unsupported public JSON competition")
        now = datetime.now(timezone.utc)
        current = now.year if now.month >= 7 else now.year - 1
        rows = []
        with httpx.Client(timeout=30, transport=httpx.HTTPTransport(retries=2), follow_redirects=True) as client:
            for season in range(start_year, current + 1):
                url = f"https://fixturedownload.com/feed/json/champions-league-{season}"
                response = client.get(url)
                response.raise_for_status()  # Never replace a full archive with a partial failed download.
                rows.extend(parse_completed_matches(response.json(), season, url, now))
                time.sleep(0.5)
        if not rows:
            raise ValueError("No completed Champions League results returned")
        df = pd.DataFrame(rows).drop_duplicates(["Date", "Home", "Away"]).sort_values(["Date", "Home"])
        df["Week"] = df.groupby(["Season", "Home"]).cumcount() + 1
        return df.reset_index(drop=True)
