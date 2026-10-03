"""Fail-closed import of public DOM evidence; model prose is never an odds feed."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
import tempfile
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse, parse_qs
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from sqlalchemy import select, func, text
from backend.app.db.models import Fixture, League, Prediction, SABookmakerOdds
from backend.app.services.betting_integrity import QUOTE_MAX_AGE, valid_odds

FEED_PATH = Path('/app/storage/bookmaker-feed/snapshot.json')
ZA = ZoneInfo('Africa/Johannesburg')


def aware(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('timezone_required')
    return parsed.astimezone(timezone.utc)


def parse_betway(source, now):
    """Only the inspected prematch 1X2 layout; reject schema drift and ambiguous dates."""
    if source.get('status') != 'collected':
        raise ValueError('source_not_collected')
    observed = aware(source['observed_at'])
    if not now - QUOTE_MAX_AGE <= observed <= now:
        raise ValueError('stale_snapshot')
    if source.get('url') != 'https://sports.betway.co.za/sport/soccer':
        raise ValueError('unexpected_source')
    headers = source.get('headers', [])
    if not isinstance(headers, list) or not all(isinstance(h, str) for h in headers):
        raise ValueError('invalid_headers')
    if not isinstance(source.get('events'), list) or not all(isinstance(e, str) for e in source['events']):
        raise ValueError('invalid_events')
    if '1X2' not in headers or not any(re.sub(r'\s+', '', h) == '1X2' for h in headers if h != '1X2'):
        raise ValueError('market_headers_changed')
    # Relative Today/Tomorrow dates are allowed only when the site's displayed
    # clock agrees with the explicitly configured ZA browser clock.
    local = observed.astimezone(ZA)
    if source.get('browser_timezone') != 'Africa/Johannesburg' or not source.get('clock'):
        raise ValueError('timezone_unconfirmed')
    clock = datetime.strptime(source['clock'], '%H:%M:%S').time()
    shown = datetime.combine(local.date(), clock, ZA)
    if abs((shown-local).total_seconds()) > 120:
        raise ValueError('source_clock_mismatch')
    records, rejected = [], Counter()
    seen = set()
    for fragment in source.get('events', [])[:500]:
        try:
            soup = BeautifulSoup(fragment, 'html.parser')
            link = soup.select_one('a[href*="eventId="]')
            if not link:
                raise ValueError('missing_event_link')
            url = urljoin(source['url'], link['href'])
            parsed = urlparse(url)
            event_id = parse_qs(parsed.query).get('eventId', [''])[0]
            if parsed.hostname != 'sports.betway.co.za' or not parsed.path.startswith('/event/soccer/') or not event_id.isdigit():
                raise ValueError('invalid_event_identity')
            if any(word in parsed.path.lower() for word in ('esoccer', 'virtual', 'simulated')):
                raise ValueError('not_real_football')
            teams = [e.get_text(strip=True) for e in link.select('strong')]
            if len(teams) != 2 or teams[0].casefold() == teams[1].casefold():
                raise ValueError('ambiguous_teams')
            schedule = re.search(r'\b(Today|Tomorrow)\s*(\d{2}:\d{2})\b', link.get_text(' ', strip=True))
            if not schedule:
                raise ValueError('ambiguous_kickoff')
            day = local.date() + timedelta(days=schedule[1] == 'Tomorrow')
            kickoff = datetime.combine(day, datetime.strptime(schedule[2], '%H:%M').time(), ZA).astimezone(timezone.utc)
            if kickoff <= now:
                raise ValueError('not_upcoming')
            prices = soup.select('[price]')
            if len(prices) < 3 or len(prices[0].parent.select('[price]')) != 3:
                raise ValueError('market_layout_changed')
            values = []
            for cell in prices[:3]:
                if cell.get('aria-disabled') == 'true' or 'disabled' in cell.attrs or cell.select('svg'):
                    raise ValueError('suspended_or_unavailable')
                value = cell.get_text(strip=True)
                if not re.fullmatch(r'\d+(?:\.\d+)?', value):
                    raise ValueError('invalid_price')
                values.append(float(value))
            if not all(valid_odds(value) and value <= 1000 for value in values):
                raise ValueError('invalid_price')
            if event_id in seen:
                continue
            seen.add(event_id)
            records.append(dict(provider_event_id=event_id, source_url=url, home_team=teams[0],
                away_team=teams[1], match_date=kickoff, scraped_at=observed,
                odds_home=values[0], odds_draw=values[1], odds_away=values[2],
                evidence_sha256=hashlib.sha256(fragment.encode()).hexdigest()))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            rejected[str(exc)] += 1
    return records, dict(rejected)


async def _import_snapshot(db, path=FEED_PATH):
    now = datetime.now(timezone.utc)
    report = {'status': 'unavailable', 'records_synced': 0, 'sources': {}, 'timestamp': now.isoformat()}
    if not path.exists():
        return {**report, 'message': 'No browser snapshot. Run sync-bookmaker-data.ps1 on the Windows host.'}
    if path.stat().st_size > 15_000_000:
        return {**report, 'message': 'Snapshot rejected: file too large.'}
    try:
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(snapshot, dict) or snapshot.get('schema_version') != 1 or snapshot.get('collector') != 'dots-dom-v1':
            raise ValueError('unsupported_snapshot')
        if not isinstance(snapshot.get('sources'), dict) or any(not isinstance(s, dict) for s in snapshot['sources'].values()):
            raise ValueError('invalid_sources')
        collected = aware(snapshot['collected_at'])
        if not now - QUOTE_MAX_AGE <= collected <= now:
            raise ValueError('stale_snapshot')
    except (OSError, ValueError, KeyError, TypeError):
        return {**report, 'message': 'Invalid or expired browser snapshot; no prices imported.'}
    await db.execute(text('SELECT pg_advisory_xact_lock(718312)'))
    for name in ('BETWAY', 'HOLLYWOODBETS'):
        source = snapshot.get('sources', {}).get(name, {})
        detail = report['sources'][name] = {'status': source.get('status', 'missing'), 'accepted': 0, 'rejected': {}}
        if name != 'BETWAY':
            detail['message'] = 'Public soccer source not returning supported event data; no fabricated fallback.'
            continue
        try:
            rows, reasons = parse_betway(source, now)
        except (ValueError, KeyError, TypeError) as exc:
            detail.update(status='rejected', reason=str(exc))
            continue
        rejected = Counter(reasons)
        for row in rows:
            candidates = (await db.execute(select(Fixture, League.name).join(League, League.id == Fixture.league_id).where(
                Fixture.is_current.is_(True), Fixture.fetched_at.between(now-timedelta(hours=48), now),
                Fixture.source_url.isnot(None), ~Fixture.source_url.ilike('%betway%'),
                func.lower(Fixture.home_team) == row['home_team'].lower(),
                func.lower(Fixture.away_team) == row['away_team'].lower(), Fixture.match_date == row['match_date']))).all()
            if len(candidates) != 1:
                rejected['no_unique_fresh_fixture_match'] += 1
                continue
            fixture, league_name = candidates[0]
            existing = (await db.execute(select(SABookmakerOdds).where(SABookmakerOdds.fixture_id == fixture.id,
                SABookmakerOdds.bookmaker == name).order_by(SABookmakerOdds.scraped_at.desc()).limit(1))).scalar_one_or_none()
            if existing and existing.scraped_at >= row['scraped_at']:
                rejected['already_imported_or_newer'] += 1
                continue
            prediction = (await db.execute(select(Prediction.id).where(Prediction.league_id == fixture.league_id,
                Prediction.home_team == fixture.home_team, Prediction.away_team == fixture.away_team,
                Prediction.match_date == fixture.match_date, Prediction.market_type == 'result')
                .order_by(Prediction.created_at.desc()).limit(1))).scalar_one_or_none()
            # Append observation history rather than destroying evidence of older prices.
            db.add(SABookmakerOdds(fixture_id=fixture.id, prediction_id=prediction, bookmaker=name,
                match_title=f'{fixture.home_team} vs {fixture.away_team}', home_team=fixture.home_team,
                away_team=fixture.away_team, league_name=league_name, match_date=fixture.match_date,
                source_url=row['source_url'], scraped_at=row['scraped_at'],
                odds_home=row['odds_home'], odds_draw=row['odds_draw'], odds_away=row['odds_away'],
                markets_data={'provenance': 'observed', 'jurisdiction': 'ZA', 'period': 'regulation',
                    'suspended': False, 'provider_event_id': row['provider_event_id'],
                    'collector': 'dots-dom-v1', 'evidence_sha256': row['evidence_sha256']}))
            detail['accepted'] += 1
            report['records_synced'] += 1
        detail['rejected'] = dict(rejected)
        detail['parsed_events'] = len(rows)
    await db.commit()
    report['status'] = 'partial' if report['records_synced'] else 'no_verified_quotes'
    report['message'] = 'Only exact, fresh fixture matches accepted. Unsupported markets and sources remain unavailable.'
    return report


async def import_snapshot(db, path=FEED_PATH):
    started = time.monotonic()
    report = None
    try:
        report = await _import_snapshot(db, path)
        return report
    except Exception as exc:
        report = dict(status='error', error_type=type(exc).__name__, records_synced=0,
                      timestamp=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        # Monitoring failures must not roll back a completed import or mask its error.
        try:
            report['duration_seconds'] = round(time.monotonic() - started, 3)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as out:
                json.dump(report, out)
            os.replace(out.name, path.parent / 'import-report.json')
            log_dir = path.parent / 'logs'
            log_dir.mkdir(exist_ok=True)
            handler = RotatingFileHandler(log_dir / 'importer.jsonl', maxBytes=2_000_000, backupCount=3, encoding='utf-8')
            try:
                entry = {**report, 'stage': 'import'}
                handler.emit(logging.LogRecord('scraper', logging.INFO, '', 0, json.dumps(entry), (), None))
            finally:
                handler.close()
        except Exception:
            logging.getLogger(__name__).warning('Could not persist scraper telemetry', exc_info=False)


async def data_quality_report(db):
    """Read-only coverage audit, not a claim that every provider fact is correct."""
    now = datetime.now(timezone.utc)
    counts = (await db.execute(text('''SELECT
        count(*) FILTER (WHERE is_current AND match_date > now()) AS upcoming,
        count(*) FILTER (WHERE is_current AND match_date > now() AND
            fetched_at BETWEEN now()-interval '48 hours' AND now() AND source_url IS NOT NULL) AS fresh_sourced,
        count(*) FILTER (WHERE is_current AND fetched_at > now()) AS future_fetch_timestamps
        FROM fixtures'''))).mappings().one()
    results = (await db.execute(text('''SELECT
        count(*) FILTER (WHERE match_date < now() AND actual_result IS NULL) AS unsettled_predictions,
        count(*) FILTER (WHERE actual_result IS NOT NULL AND
            (actual_score IS NULL OR result_source IS NULL OR result_verified_at IS NULL)) AS missing_result_evidence
        FROM predictions'''))).mappings().one()
    sources = {}
    try:
        if FEED_PATH.stat().st_size > 15_000_000:
            raise ValueError('oversized')
        snapshot = json.loads(FEED_PATH.read_text(encoding='utf-8'))
        for name in ('BETWAY', 'HOLLYWOODBETS'):
            source = snapshot.get('sources', {}).get(name, {})
            detail = sources[name] = {'status': source.get('status', 'missing'),
                'observed_at': source.get('observed_at'), 'event_blocks': len(source.get('events', []))}
            if name == 'BETWAY':
                try:
                    rows, rejected = parse_betway(source, now)
                    detail.update(parsed_events=len(rows), rejected=rejected)
                except (ValueError, KeyError, TypeError) as exc:
                    detail.update(status='rejected', reason=str(exc))
    except (OSError, ValueError, TypeError, AttributeError):
        sources = {'status': 'snapshot_unavailable_or_invalid'}
    return {'timestamp': now.isoformat(), 'fixtures': dict(counts), 'results': dict(results),
        'sources': sources, 'coverage': 'partial',
        'note': 'Source and freshness checks are not independent verification of every fact. Only exact fixture-matched quotes are published.'}


if __name__ == '__main__':
    import asyncio
    from backend.app.db.session import async_session_factory

    async def main():
        async with async_session_factory() as db:
            print(json.dumps(await import_snapshot(db)))

    asyncio.run(main())
