"""Decimal-odds validation and regulation-time journal settlement."""
import math
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from urllib.parse import urlparse

MARKETS = {"H": "odds_home", "D": "odds_draw", "A": "odds_away",
    "Over 1.5": "over_15", "Under 1.5": "under_15", "Over 2.5": "over_25",
    "Under 2.5": "under_25", "Over 3.5": "over_35", "Under 3.5": "under_35",
    "BTTS Yes": "btts_yes", "BTTS No": "btts_no", "1X": "dc_1x", "12": "dc_12",
    "X2": "dc_x2", "DNB 1": "dnb_home", "DNB 2": "dnb_away"}
BOOK_HOSTS = {"HOLLYWOODBETS": "hollywoodbets.net", "BETWAY": "betway.co.za"}
QUOTE_MAX_AGE = timedelta(minutes=15)


def valid_odds(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 1


def money(value):
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def normalize_selection(value):
    pick = re.sub(r"\s*\((?:HOLLYWOODBETS|BETWAY(?: SOUTH AFRICA)?|SPORTINGBET|SUPABETS)\)\s*$", "", value.strip(), flags=re.I)
    upper = pick.upper()
    aliases = {"1": "H", "HOME": "H", "HOME WIN (1)": "H", "HOME (1)": "H",
        "X": "D", "DRAW": "D", "DRAW (X)": "D", "2": "A", "AWAY": "A",
        "AWAY WIN (2)": "A", "AWAY (2)": "A", "OVER": "Over 2.5", "UNDER": "Under 2.5",
        "O2.5": "Over 2.5", "U2.5": "Under 2.5",
        "DOUBLE CHANCE (1X)": "1X", "DOUBLE CHANCE 1X": "1X", "DOUBLE CHANCE (X2)": "X2",
        "DOUBLE CHANCE (12)": "12", "DRAW NO BET (HOME)": "DNB 1", "DRAW NO BET 1": "DNB 1",
        "DRAW NO BET (AWAY)": "DNB 2", "DRAW NO BET 2": "DNB 2"}
    upper = upper.removesuffix(" GOALS")
    if upper in aliases:
        return aliases[upper]
    from backend.app.services.prediction_markets import SELECTION_TARGETS
    for canonical in set(MARKETS) | set(SELECTION_TARGETS):
        if canonical.upper() == upper:
            return canonical
    raise ValueError("Unsupported market. Use a supported regulation-time selection.")


def settlement(selection, actual, score):
    try:
        selection = normalize_selection(selection)
    except (ValueError, AttributeError):
        return None
    parsed = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", score or "")
    if not parsed:
        return None
    home, away = int(parsed[1]), int(parsed[2])
    outcome = "H" if home > away else "A" if away > home else "D"
    if outcome != actual:
        return None
    if selection.startswith(("Corners ", "Shots on target ")):
        return None  # Goal scores cannot settle corners/shots.
    if selection.startswith(("Home goals ", "Away goals ")):
        direction, line = selection.split()[-2:]
        count = home if selection.startswith("Home") else away
        won = count > float(line) if direction == "Over" else count < float(line)
    elif selection.startswith("DNB"):
        if outcome == "D":
            return "VOID"
        won = outcome == ("H" if selection == "DNB 1" else "A")
    elif selection in ("1X", "12", "X2"):
        won = {"H": "1", "D": "X", "A": "2"}[outcome] in selection
    elif selection.startswith("BTTS"):
        won = (home > 0 and away > 0) == (selection == "BTTS Yes")
    elif selection.startswith(("Over", "Under")):
        direction, line = selection.split()
        won = home + away > float(line) if direction == "Over" else home + away < float(line)
    else:
        won = selection == outcome
    return "WON" if won else "LOST"


def settle_legs(legs, stake):
    if not legs:
        return "PENDING", 0.0
    states = [leg.get("status", "PENDING") for leg in legs]
    if "LOST" in states:
        return "LOST", money(-stake)
    if any(state not in ("WON", "VOID") for state in states):
        return "PENDING", 0.0
    if all(state == "VOID" for state in states):
        return "VOID", 0.0
    multiplier = math.prod(leg["odds_taken"] for leg in legs if leg["status"] == "WON")
    return "WON", money(stake * (multiplier - 1))


def observed_quote(row, fixture=None, now=None):
    """Legacy/model-generated rows cannot be promoted to observed prices."""
    now = now or datetime.now(timezone.utc)
    meta = row.markets_data or {}
    host = urlparse(row.source_url or "").hostname or ""
    expected = BOOK_HOSTS.get(row.bookmaker)
    if not expected or not (host == expected or host.endswith("." + expected)):
        return False
    if urlparse(row.source_url or "").scheme != "https" or not meta.get("provider_event_id"):
        return False
    if meta.get("provenance") != "observed" or meta.get("jurisdiction") != "ZA" or meta.get("period") != "regulation":
        return False
    if meta.get("suspended") is not False or row.scraped_at is None or row.match_date is None:
        return False
    if row.scraped_at.tzinfo is None or row.match_date.tzinfo is None:
        return False
    if not now - QUOTE_MAX_AGE <= row.scraped_at <= now or row.match_date <= now:
        return False
    if fixture is not None:
        return bool(fixture.is_current and fixture.source_url and fixture.fetched_at
            and fixture.fetched_at.tzinfo is not None
            and now - timedelta(hours=48) <= fixture.fetched_at <= now
            and fixture.home_team == row.home_team and fixture.away_team == row.away_team
            and fixture.match_date == row.match_date)
    return True
