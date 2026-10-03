import asyncio
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from lxml import html as lxml_html

from backend.app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Rate limiting configuration
MIN_REQUEST_DELAY_SECONDS = 2.0  # Minimum delay between requests
MAX_RETRIES = 3  # Maximum retries for rate-limited requests
RETRY_BACKOFF_FACTOR = 2.0  # Exponential backoff factor
MAX_CONCURRENT_REQUESTS = 3  # Maximum concurrent HTTP requests


def _get_sync_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.config import get_settings
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL_SYNC)
    return sessionmaker(bind=engine)()


# Async HTTP client with connection pooling
_async_client: Optional[httpx.AsyncClient] = None
_request_semaphore: Optional[asyncio.Semaphore] = None
_espn_forbidden = False


async def _get_async_client() -> httpx.AsyncClient:
    """Get or create async HTTP client with connection pooling."""
    global _async_client, _request_semaphore
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=10.0),
            transport=httpx.AsyncHTTPTransport(retries=2),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=True,
        )
        _request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _async_client


async def _close_async_client():
    """Close async HTTP client on shutdown."""
    global _async_client, _request_semaphore, _espn_forbidden
    _espn_forbidden = False
    if _async_client:
        await _async_client.aclose()
        _async_client = None
        _request_semaphore = None


def _normalize_fixture_url(url: str) -> str:
    """Fix common catalog typos (e.g. missing leading 'h' in https)."""
    u = (url or "").strip()
    if u.startswith("ttps://"):
        return "h" + u
    return u


ESPN_LEAGUES = {
    ("Europe", "Champions-League"): "uefa.champions",
    ("Argentina", "Primera-Division"): "arg.1",
    ("Belgium", "Jupiler-League"): "bel.1",
    ("Brazil", "Serie-A"): "bra.1",
    ("China", "Super-League"): "chn.1",
    ("Denmark", "Super-Liga"): "den.1",
    ("England", "League-1"): "eng.3",
    ("England", "League-2"): "eng.4",
    ("Finland", "Veikkausliiga"): "fin.1",
    ("France", "Ligue-2"): "fra.2",
    ("Germany", "Bundesliga-2"): "ger.2",
    ("Greece", "Super-League"): "gre.1",
    ("Ireland", "Premier-Division"): "irl.1",
    ("Italy", "Serie-B"): "ita.2",
    ("Japan", "J-1"): "jpn.1",
    ("Mexico", "Liga-MX"): "mex.1",
    ("Norway", "Eliteserien"): "nor.1",
    ("Romania", "Liga-1"): "rou.1",
    ("Russia", "Premier-League"): "rus.1",
    ("Scotland", "Premiership"): "sco.1",
    ("Spain", "Segunda-Division"): "esp.2",
    ("Sweden", "Allsvenskan"): "swe.1",
    ("Switzerland", "Super-League"): "sui.1",
    ("Turkey", "Super-Lig"): "tur.1",
    ("England", "Premier-League"): "eng.1",
    ("England", "Championship"): "eng.2",
    ("Spain", "La-Liga"): "esp.1",
    ("Germany", "Bundesliga-1"): "ger.1",
    ("Italy", "Serie-A"): "ita.1",
    ("France", "Ligue-1"): "fra.1",
    ("Netherlands", "Eredivisie"): "ned.1",
    ("Portugal", "Liga-1"): "por.1",
    ("USA", "MLS"): "usa.1",
    ("South Africa", "Betway-Premiership"): "rsa.1",
}

FOOTBALL_DATA_COMPETITIONS = {
    ("Europe", "Champions-League"): "CL",
    ("England", "Premier-League"): "PL",
    ("England", "Championship"): "ELC",
    ("Spain", "La-Liga"): "PD",
    ("Germany", "Bundesliga-1"): "BL1",
    ("Italy", "Serie-A"): "SA",
    ("France", "Ligue-1"): "FL1",
    ("Netherlands", "Eredivisie"): "DED",
    ("Portugal", "Liga-1"): "PPL",
}

API_FOOTBALL_LEAGUES = {
    ("Europe", "Champions-League"): 2,
    ("England", "Premier-League"): 39,
    ("England", "Championship"): 40,
    ("Spain", "La-Liga"): 140,
    ("Germany", "Bundesliga-1"): 78,
    ("Italy", "Serie-A"): 135,
    ("France", "Ligue-1"): 61,
    ("Netherlands", "Eredivisie"): 88,
    ("Portugal", "Liga-1"): 94,
    ("USA", "MLS"): 253,
    ("South Africa", "Betway-Premiership"): 288,
}

THESPORTSDB_LEAGUES = {
    ("England", "Premier-League"): 4328,
    ("England", "Championship"): 4329,
    ("Spain", "La-Liga"): 4335,
    ("Germany", "Bundesliga-1"): 4331,
    ("Italy", "Serie-A"): 4332,
    ("France", "Ligue-1"): 4334,
    ("Netherlands", "Eredivisie"): 4337,
    ("Portugal", "Liga-1"): 4344,
    ("USA", "MLS"): 4346,
    ("South Africa", "Betway-Premiership"): 4802,
}

FIXTURE_DOWNLOAD_LEAGUES = {
    ("Europe", "Champions-League"): "champions-league",
    ("England", "League-1"): "efl-league-one",
    ("England", "League-2"): "efl-league-two",
    ("France", "Ligue-2"): "ligue-2",
    ("Scotland", "Premiership"): "scottish-premiership",
    ("Turkey", "Super-Lig"): "super-lig",
    ("England", "Premier-League"): "epl",
    ("England", "Championship"): "championship",
    ("Spain", "La-Liga"): "la-liga",
    ("Germany", "Bundesliga-1"): "bundesliga",
    ("Italy", "Serie-A"): "serie-a",
    ("France", "Ligue-1"): "ligue-1",
    ("Netherlands", "Eredivisie"): "eredivisie",
    ("Portugal", "Liga-1"): "primeira-liga",
    ("USA", "MLS"): "mls",
}


async def _fetch_fixture_download_fixtures(league) -> list[dict]:
    """Scrape Fixture Download's public season JSON feed."""
    slug = FIXTURE_DOWNLOAD_LEAGUES.get((league.country, league.name))
    if not slug:
        return []
    now = datetime.now(timezone.utc)
    season = now.year if now.month >= 7 or slug == "mls" else now.year - 1
    url = f"https://fixturedownload.com/feed/json/{slug}-{season}"
    try:
        client = await _get_async_client()
        async with _request_semaphore:
            response = await client.get(url, headers={"User-Agent": "ProphitBet/1.0", "Accept": "application/json"})
        response.raise_for_status()
        fixtures = []
        for match in response.json():
            home = match.get("HomeTeam")
            away = match.get("AwayTeam")
            raw_date = match.get("DateUtc")
            if not (home and away and raw_date):
                continue
            match_date = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
            if match_date >= now and match.get("HomeTeamScore") is None and match.get("AwayTeamScore") is None:
                fixtures.append({
                    "home_team": home,
                    "away_team": away,
                    "match_date": match_date,
                    "source_url": url,
                })
        return fixtures
    except Exception as exc:
        logger.warning("Fixture Download failed for %s - %s: %s", league.country, league.name, exc)
        return []


async def _fetch_thesportsdb_fixtures(league) -> list[dict]:
    """Use TheSportsDB's documented free v1 schedule as a final factual fallback."""
    league_id = THESPORTSDB_LEAGUES.get((league.country, league.name))
    if not league_id:
        return []
    url = f"https://www.thesportsdb.com/api/v1/json/123/eventsnextleague.php?id={league_id}"
    now = datetime.now(timezone.utc)
    try:
        client = await _get_async_client()
        async with _request_semaphore:
            response = await client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        fixtures = []
        for event in response.json().get("events") or []:
            if event.get("strStatus") in {"Match Finished", "FT", "Postponed", "Cancelled"}:
                continue
            home = event.get("strHomeTeam")
            away = event.get("strAwayTeam")
            timestamp = event.get("strTimestamp")
            if not (home and away and timestamp):
                continue
            match_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if match_date.tzinfo is None:
                match_date = match_date.replace(tzinfo=timezone.utc)
            if match_date >= now:
                fixtures.append({
                    "home_team": home,
                    "away_team": away,
                    "match_date": match_date,
                    "source_url": url,
                })
        return fixtures
    except Exception as exc:
        logger.warning("TheSportsDB failed for %s - %s: %s", league.country, league.name, exc)
        return []


def _api_football_season(now: datetime, league_id: int) -> int:
    # Calendar-year leagues such as MLS use the current year. European and
    # South African competitions use the year in which their season starts.
    if league_id == 253:
        return now.year
    return now.year if now.month >= 7 else now.year - 1


async def _fetch_api_football_fixtures(league) -> list[dict]:
    """Load factual fixtures and observed 1X2 odds from API-Football."""
    from backend.app.config import get_settings

    token = get_settings().API_FOOTBALL_KEY.strip()
    league_id = API_FOOTBALL_LEAGUES.get((league.country, league.name))
    if not token or not league_id:
        return []
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=30)
    base_url = "https://v3.football.api-sports.io"
    headers = {"x-apisports-key": token, "Accept": "application/json"}
    params = {
        "league": league_id,
        "season": _api_football_season(now, league_id),
        "from": now.date().isoformat(),
        "to": end.date().isoformat(),
        "timezone": "UTC",
    }
    try:
        client = await _get_async_client()
        async with _request_semaphore:
            response = await client.get(f"{base_url}/fixtures", params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise ValueError(str(payload["errors"]))

        # The odds endpoint is requested once per league, not once per match,
        # to stay within the free daily quota. Only complete Match Winner lines
        # from a named bookmaker are accepted.
        odds_by_fixture: dict[int, tuple[float, float, float]] = {}
        odds_params = {"league": league_id, "season": params["season"], "page": 1}
        async with _request_semaphore:
            odds_response = await client.get(f"{base_url}/odds", params=odds_params, headers=headers)
        if odds_response.is_success:
            odds_payload = odds_response.json()
            if not odds_payload.get("errors"):
                for item in odds_payload.get("response", []):
                    fixture_id = (item.get("fixture") or {}).get("id")
                    for bookmaker in item.get("bookmakers", []):
                        market = next((b for b in bookmaker.get("bets", []) if b.get("name") == "Match Winner"), None)
                        if not market:
                            continue
                        values = {str(v.get("value")): v.get("odd") for v in market.get("values", [])}
                        try:
                            line = (float(values["Home"]), float(values["Draw"]), float(values["Away"]))
                        except (KeyError, TypeError, ValueError):
                            continue
                        if all(price > 1.0 for price in line):
                            odds_by_fixture[int(fixture_id)] = line
                            break

        fixtures = []
        for item in payload.get("response", []):
            meta = item.get("fixture") or {}
            if (meta.get("status") or {}).get("short") != "NS":
                continue
            teams = item.get("teams") or {}
            fixture_id = meta.get("id")
            utc_date = meta.get("date")
            home = (teams.get("home") or {}).get("name")
            away = (teams.get("away") or {}).get("name")
            if not (fixture_id and utc_date and home and away):
                continue
            match_date = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
            if match_date < now:
                continue
            row = {
                "home_team": home,
                "away_team": away,
                "match_date": match_date,
                "source_url": str(response.url),
            }
            if int(fixture_id) in odds_by_fixture:
                row["odds_1"], row["odds_x"], row["odds_2"] = odds_by_fixture[int(fixture_id)]
            fixtures.append(row)
        return fixtures
    except Exception as exc:
        logger.warning("API-Football failed for %s - %s: %s", league.country, league.name, exc)
        return []


async def _fetch_football_data_fixtures(league) -> list[dict]:
    """Load scheduled fixtures from football-data.org when a token is configured."""
    from backend.app.config import get_settings

    token = get_settings().FOOTBALL_DATA_API_KEY.strip()
    competition = FOOTBALL_DATA_COMPETITIONS.get((league.country, league.name))
    if not token or not competition:
        return []
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=30)
    url = f"https://api.football-data.org/v4/competitions/{competition}/matches"
    try:
        client = await _get_async_client()
        async with _request_semaphore:
            response = await client.get(
                url,
                params={"dateFrom": now.date().isoformat(), "dateTo": end.date().isoformat(), "status": "SCHEDULED,TIMED"},
                headers={"X-Auth-Token": token, "Accept": "application/json"},
            )
        response.raise_for_status()
        fixtures = []
        for match in response.json().get("matches", []):
            if match.get("status") != "TIMED":
                continue
            home = (match.get("homeTeam") or {}).get("name")
            away = (match.get("awayTeam") or {}).get("name")
            utc_date = match.get("utcDate")
            if not (home and away and utc_date):
                continue
            match_date = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
            if match_date >= now:
                fixtures.append({
                    "home_team": home,
                    "away_team": away,
                    "match_date": match_date,
                    "source_url": str(response.url),
                })
        return fixtures
    except Exception as exc:
        logger.warning("football-data.org failed for %s - %s: %s", league.country, league.name, exc)
        return []


async def _scrape_espn_fixtures(league) -> list[dict]:
    """Load scheduled fixtures from ESPN's public scoreboard JSON feed."""
    global _espn_forbidden
    if _espn_forbidden:
        return []
    league_code = ESPN_LEAGUES.get((league.country, league.name))
    if not league_code:
        return []
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=30)
    # Soccer rejects YYYYMMDD-YYYYMMDD ranges with HTTP 400. Query calendar
    # months, then apply an exact UTC window locally (including year rollover).
    month = now.replace(day=1)
    try:
        client = await _get_async_client()
        fixtures = []
        while month.date() <= end.date():
            url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard"
                   f"?dates={month:%Y%m}&limit=1000")
            async with _request_semaphore:
                if _espn_forbidden:
                    return []
                response = await client.get(url, headers={"User-Agent": "ProphitBet/1.0"})
                if response.status_code == 403:
                    _espn_forbidden = True
                    logger.warning("ESPN denied scraper access; disabling this provider for the rest of this sync")
                    return []
                # Bound aggregate traffic, including empty or failing feeds.
                await asyncio.sleep(1)
            response.raise_for_status()
            payload = response.json()
            if not any(item.get("slug") == league_code for item in payload.get("leagues", [])):
                raise ValueError("Provider returned a different competition")
            if len(payload.get("events", [])) >= 1000:
                raise ValueError("Provider schedule may be truncated")
            fixtures.extend(_parse_espn_events(payload.get("events", []), url, now, end))
            month = (month + timedelta(days=32)).replace(day=1)
        return valid_fixture_rows(fixtures, now)
    except Exception as exc:
        logger.warning("ESPN fixture feed failed for %s - %s: %s", league.country, league.name, exc)
        return []


def _parse_espn_events(events, source_url, now, end):
    fixtures = []
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            if event.get("status", {}).get("type", {}).get("name") != "STATUS_SCHEDULED":
                continue
            competition = (event.get("competitions") or [{}])[0]
            if competition.get("timeValid") is not True:
                continue
            teams = {entry.get("homeAway"): entry.get("team", {}).get("displayName")
                     for entry in competition.get("competitors", [])}
            date = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            if date.tzinfo is None or not now <= date <= end:
                continue
            fixtures.append(dict(home_team=teams.get("home"), away_team=teams.get("away"),
                                 match_date=date, source_url=source_url))
        except (KeyError, TypeError, ValueError, AttributeError):
            logger.warning("Skipping malformed ESPN event %s", event.get("id"))
    return valid_fixture_rows(fixtures, now)


async def _scrape_league_fixtures_async(league) -> list[dict]:
    fixtures = await _fetch_football_data_fixtures(league)
    fixtures = fixtures or await _fetch_fixture_download_fixtures(league)
    fixtures = fixtures or await _scrape_espn_fixtures(league)
    fixtures = fixtures or await _fetch_api_football_fixtures(league)
    fixtures = fixtures or await _fetch_thesportsdb_fixtures(league)
    # HTML date headers omit kickoff times and sometimes years/time zones.
    # They cannot substantiate an exact UTC kickoff and must not be published.
    return fixtures


def _parse_footystats_date_header(header: str) -> datetime | None:
    """
    Parse FootyStats h2 text. Often short month/day only, e.g. 'May 24 ~' (no year),
    or full '09 Apr 2026 ~'.
    """
    s = (header or "").replace("~", "").strip()
    s = re.sub(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*", "", s, flags=re.I)
    for fmt in ("%d %b %Y", "%d %B %Y", "%d %b %y", "%d %B %y"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    now = datetime.now(timezone.utc)
    for fmt in ("%b %d", "%B %d", "%d %b", "%d %B"):
        try:
            base = datetime.strptime(s.strip(), fmt)
            dt = base.replace(tzinfo=timezone.utc).replace(year=now.year)
            if dt < now - timedelta(days=180):
                dt = dt.replace(year=now.year + 1)
            return dt
        except ValueError:
            continue

    return None


def _scrape_footystats_fixtures(html: str) -> list[dict]:
    """
    FootyStats uses div.full-matches-table blocks with h2 date headers and ul rows
    (same structure as src/network/fixtures/footystats/scraper.py).
    """
    fixtures: list[dict] = []
    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        return fixtures

    for div in tree.xpath('//div[contains(@class, "full-matches-table")]'):
        h2_el = div.xpath(".//h2")
        if not h2_el:
            continue
        header_text = h2_el[0].text_content() if h2_el[0].text_content() else ""
        match_date = _parse_footystats_date_header(header_text)

        uls = div.xpath(".//ul")
        if len(uls) < 2:
            continue
        for ul in uls[1:]:
            try:
                links = ul.xpath(".//a")
                if len(links) < 3:
                    continue
                home_spans = links[0].xpath(".//span")
                away_spans = links[2].xpath(".//span")
                if not home_spans or not away_spans:
                    continue
                home = home_spans[0].text_content().strip()
                away = away_spans[0].text_content().strip()
                if not home or not away:
                    continue
                fixtures.append({
                    "home_team": home,
                    "away_team": away,
                    "match_date": match_date,
                })
            except (IndexError, AttributeError):
                continue

    return fixtures


def _scrape_legacy_table_fixtures(html: str) -> list[dict]:
    """Fallback: simple HTML tables used by some older fixture pages."""
    fixtures: list[dict] = []
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("table tr, .fixture-row, .match-row")
    for row in rows:
        cells = row.find_all("td") if row.name == "tr" else row.find_all(class_=True)
        if len(cells) < 3:
            continue
        try:
            text_parts = [c.get_text(strip=True) for c in cells]
            home = text_parts[0] if text_parts else ""
            away = text_parts[2] if len(text_parts) > 2 else ""
            date_str = text_parts[-1] if text_parts else ""
            if not home or not away:
                continue
            match_date = None
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d-%m-%Y"):
                try:
                    match_date = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
            fixtures.append({"home_team": home, "away_team": away, "match_date": match_date})
        except (IndexError, ValueError):
            continue
    return fixtures


# Module-level variable to track last request time for rate limiting
_last_request_time = 0.0


def _scrape_fixtures_from_url(url: str) -> list[dict]:
    """
    Fetch fixture pages (FootyStats + legacy table layouts) with rate limiting and retries.
    """
    url = _normalize_fixture_url(url)
    if not url or not url.startswith("http"):
        return []

    global _last_request_time
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    for attempt in range(MAX_RETRIES):
        try:
            # Rate limiting: ensure minimum delay between requests
            elapsed = time.time() - _last_request_time
            if elapsed < MIN_REQUEST_DELAY_SECONDS:
                sleep_time = MIN_REQUEST_DELAY_SECONDS - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s before request to {url}")
                time.sleep(sleep_time)

            _last_request_time = time.time()
            response = httpx.get(url, headers=headers, timeout=45, follow_redirects=True)
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        wait_time = MIN_REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** attempt)
                else:
                    wait_time = MIN_REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** attempt)
                
                logger.warning(f"Rate limited (429) for {url}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait_time)
                continue
            if response.status_code == 403:
                logger.warning("Fixture source denied automated access (403): %s", url)
                return []
            
            response.raise_for_status()
            html = response.text

            fixtures = _scrape_footystats_fixtures(html)
            if not fixtures:
                fixtures = _scrape_legacy_table_fixtures(html)

            if not fixtures:
                logger.warning("No fixtures parsed from %s (page structure may have changed)", url)
            return fixtures

        except httpx.HTTPStatusError as e:
            # Authentication/anti-bot failures are deterministic; retrying only
            # delays the fallback and increases load on the upstream site.
            logger.warning("Scraping failed for %s: %s", url, e)
            if 400 <= e.response.status_code < 500:
                return []
            if attempt >= MAX_RETRIES - 1:
                return []
            time.sleep(MIN_REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** attempt))
        except httpx.TimeoutException:
            logger.warning(f"Timeout scraping {url}, attempt {attempt + 1}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(MIN_REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** attempt))
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(MIN_REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** attempt))
            else:
                return []

    logger.error(f"All {MAX_RETRIES} attempts failed for {url}")
    return []


# Async rate limiter using token bucket pattern
class AsyncRateLimiter:
    """Async rate limiter for HTTP requests."""
    def __init__(self, min_delay: float = MIN_REQUEST_DELAY_SECONDS):
        self.min_delay = min_delay
        self._last_request = 0.0
        self._lock = asyncio.Lock()
    
    async def wait(self):
        async with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self.min_delay:
                await asyncio.sleep(self.min_delay - elapsed)
            self._last_request = time.time()


_rate_limiter = AsyncRateLimiter()


async def _scrape_fixtures_from_url_async(url: str) -> list[dict]:
    """
    Async version: Fetch fixture pages with rate limiting and retries.
    Uses connection pooling and semaphore for concurrency control.
    """
    url = _normalize_fixture_url(url)
    if not url or not url.startswith("http"):
        return []

    client = await _get_async_client()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    for attempt in range(MAX_RETRIES):
        try:
            # Rate limiting
            await _rate_limiter.wait()
            
            # Use semaphore to limit concurrent requests
            async with _request_semaphore:
                response = await client.get(url, headers=headers)
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        wait_time = MIN_REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** attempt)
                else:
                    wait_time = MIN_REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** attempt)
                
                logger.warning(f"Rate limited (429) for {url}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                await asyncio.sleep(wait_time)
                continue
            if response.status_code == 403:
                logger.warning("Fixture source denied automated access (403): %s", url)
                return []
            
            response.raise_for_status()
            html = response.text

            fixtures = _scrape_footystats_fixtures(html)
            if not fixtures:
                fixtures = _scrape_legacy_table_fixtures(html)

            if not fixtures:
                logger.warning("No fixtures parsed from %s (page structure may have changed)", url)
            return fixtures

        except httpx.TimeoutException:
            logger.warning(f"Timeout scraping {url}, attempt {attempt + 1}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(MIN_REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** attempt))
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(MIN_REQUEST_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** attempt))
            else:
                return []

    logger.error(f"All {MAX_RETRIES} attempts failed for {url}")
    return []


async def _scrape_all_leagues_async(leagues: list) -> int:
    """
    Scrape fixtures for all leagues concurrently.
    Returns total fixtures scraped.
    """
    tasks = [_scrape_league_fixtures_async(lg) for lg in leagues]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_fixtures = 0
    for lg, result in zip(leagues, results):
        if isinstance(result, Exception):
            logger.error(f"Fixture scrape failed for {lg.country} - {lg.name}: {result}")
            continue
        scraped = [fixture for fixture in result if fixture.get("match_date") is not None]
        scraped = [fixture for fixture in scraped if fixture.get("match_date") is not None]
        if not scraped:
            continue
        total_fixtures += len(scraped)
        logger.info(f"Scraped {len(scraped)} fixtures for {lg.country} - {lg.name}")
    
    return total_fixtures


@celery_app.task(name="backend.app.workers.fixtures.scrape_all_fixtures_task", bind=True)
def scrape_all_fixtures_task(self):
    """Scrape upcoming fixtures for all active leagues."""
    session = _get_sync_session()
    try:
        from backend.app.db.models import Fixture, League

        leagues = session.query(League).filter(
            League.is_active == True,
        ).all()
        # Serialize refreshes in PostgreSQL so a scheduled task and an admin
        # click cannot publish competing snapshots.
        from sqlalchemy import text
        session.execute(text("SELECT pg_advisory_xact_lock(718309)"))
        self.update_state(state="PROGRESS", meta={
            "current": 0, "total": len(leagues), "phase": "Fetching fixture providers", "item": None,
        })

        # Run async scraping
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # First, scrape all fixtures concurrently
            scraped_by_league = loop.run_until_complete(_scrape_all_leagues_async_with_data(leagues, self))
            
            total_fixtures = 0
            observed_fixtures = 0
            refreshed = 0
            unavailable = []
            for index, (lg, scraped) in enumerate(scraped_by_league, start=1):
                item = f"{lg.country} — {lg.name}"
                self.update_state(state="PROGRESS", meta={
                    "current": len(leagues) + index - 1, "total": len(leagues) * 2, "phase": "Saving sourced fixtures", "item": item,
                })
                now = datetime.now(timezone.utc)
                scraped = valid_fixture_rows(scraped, now)
                if not scraped:
                    unavailable.append(item)
                    self.update_state(state="PROGRESS", meta={
                        "current": len(leagues) + index, "total": len(leagues) * 2, "phase": "No current schedule available", "item": item,
                    })
                    continue

                # Retain old rows and provenance, but publish only the latest
                # observed schedule. This handles rescheduling, cancellations,
                # provider name changes and old duplicates without deleting data.
                session.query(Fixture).filter(Fixture.league_id == lg.id).update(
                    {Fixture.is_current: False}, synchronize_session="fetch"
                )
                refreshed += 1
                observed_fixtures += len(scraped)

                for fix in scraped:
                    existing = (
                        session.query(Fixture)
                        .filter(
                            Fixture.league_id == lg.id,
                            Fixture.home_team == fix["home_team"],
                            Fixture.away_team == fix["away_team"],
                            Fixture.match_date == fix["match_date"],
                        )
                        .first()
                    )
                    if existing:
                        existing.is_current = True
                        # Refresh only fields directly observed in the latest
                        # provider response; never overwrite them with defaults.
                        existing.source_url = fix.get("source_url") or existing.source_url
                        existing.fetched_at = datetime.now(timezone.utc)
                        if all(fix.get(key) is not None for key in ("odds_1", "odds_x", "odds_2")):
                            existing.odds_1 = fix["odds_1"]
                            existing.odds_x = fix["odds_x"]
                            existing.odds_2 = fix["odds_2"]
                        else:
                            # Odds are time-sensitive and must come from the
                            # current provider response. Do not preserve stale
                            # legacy values when the provider supplied none.
                            existing.odds_1 = None
                            existing.odds_x = None
                            existing.odds_2 = None
                        continue

                    db_fix = Fixture(
                        is_current=True,
                        league_id=lg.id,
                        home_team=fix["home_team"],
                        away_team=fix["away_team"],
                        match_date=fix["match_date"],
                        odds_1=fix.get("odds_1"),
                        odds_x=fix.get("odds_x"),
                        odds_2=fix.get("odds_2"),
                        source_url=fix.get("source_url") or _normalize_fixture_url(lg.fixture_url or ""),
                        fetched_at=datetime.now(timezone.utc),
                    )
                    session.add(db_fix)
                    total_fixtures += 1

                session.flush()
                logger.info(f"Scraped {len(scraped)} fixtures for {lg.country} - {lg.name}")
                self.update_state(state="PROGRESS", meta={
                    "current": len(leagues) + index, "total": len(leagues) * 2, "phase": "League fixtures saved", "item": item,
                })

            if not refreshed:
                raise RuntimeError("No provider returned a usable upcoming schedule. Existing data was retained; check provider availability and API keys.")
            session.commit()
            return {"total_fixtures": observed_fixtures, "new_fixtures": total_fixtures,
                    "coverage": "partial" if unavailable else "complete", "leagues_refreshed": refreshed,
                    "leagues_unavailable": unavailable, "total": len(leagues)}
        finally:
            loop.run_until_complete(_close_async_client())
            loop.close()
    finally:
        session.close()
        # Invalidate fixtures cache after scraping
        try:
            from backend.app.services.cache import invalidate_fixtures_cache
            invalidate_fixtures_cache()
        except Exception as e:
            logger.warning(f"Failed to invalidate fixtures cache: {e}")


def valid_fixture_rows(rows: list[dict], now: datetime) -> list[dict]:
    """Accept only sourced, timezone-aware future kickoffs; deduplicate response."""
    unique = {}
    for row in rows:
        date = row.get("match_date")
        home, away = (row.get("home_team") or "").strip(), (row.get("away_team") or "").strip()
        if not isinstance(date, datetime) or date.tzinfo is None or date.utcoffset() is None:
            continue
        if not home or not away or home.casefold() == away.casefold() or not row.get("source_url"):
            continue
        date = date.astimezone(timezone.utc)
        if not now <= date <= now + timedelta(days=370):
            continue
        unique[(home.casefold(), away.casefold(), date)] = {**row, "home_team": home, "away_team": away, "match_date": date}
    return list(unique.values())


async def _scrape_all_leagues_async_with_data(leagues: list, task=None) -> list[tuple]:
    """
    Scrape fixtures for all leagues concurrently.
    Returns list of (league, fixtures) tuples.
    """
    completed = 0
    async def fetch(lg):
        nonlocal completed
        try:
            return await _scrape_league_fixtures_async(lg)
        finally:
            completed += 1
            if task:
                task.update_state(state="PROGRESS", meta={
                    "current": completed, "total": len(leagues) * 2,
                    "phase": "Fetching fixture providers", "item": f"{lg.country} — {lg.name}",
                })
    tasks = [fetch(lg) for lg in leagues]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    result_list = []
    for lg, result in zip(leagues, results):
        if isinstance(result, Exception):
            logger.error(f"Fixture scrape failed for {lg.country} - {lg.name}: {result}")
            result_list.append((lg, []))
            continue
        scraped = result
        result_list.append((lg, scraped))
        if scraped:
            logger.info(f"Scraped {len(scraped)} fixtures for {lg.country} - {lg.name}")
    
    return result_list
