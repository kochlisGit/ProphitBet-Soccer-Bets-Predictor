"""Bounded-label, read-only metrics from the host collector and importer."""
import json
from datetime import datetime
from pathlib import Path
from prometheus_client.core import GaugeMetricFamily

ROOT = Path('/app/storage/bookmaker-feed')
STATUSES = {'collected', 'no_events', 'collection_failed', 'unsupported_markup',
            'robots_disallowed', 'unavailable', 'partial', 'no_verified_quotes', 'error', 'rejected'}
REASONS = {'suspended_or_unavailable', 'ambiguous_teams', 'market_layout_changed',
           'not_real_football', 'no_unique_fresh_fixture_match', 'already_imported_or_newer',
           'missing_event_link', 'invalid_event_identity', 'ambiguous_kickoff',
           'not_upcoming', 'invalid_price'}


def scraper_metrics(root=ROOT):
    health = GaugeMetricFamily('prophitbet_scraper_telemetry_available', 'Telemetry file is readable and valid (not source health)', labels=['stage'])
    timestamp = GaugeMetricFamily('prophitbet_scraper_timestamp_seconds', 'Last recorded attempt time; use age to detect stale data', labels=['stage'])
    duration = GaugeMetricFamily('prophitbet_scraper_duration_seconds', 'Duration of the last completed attempt', labels=['stage'])
    status = GaugeMetricFamily('prophitbet_scraper_status', 'Last recorded status, not current availability', labels=['stage', 'bookmaker', 'status'])
    records = GaugeMetricFamily('prophitbet_scraper_records', 'Record counts from the last attempt, not cumulative', labels=['stage', 'bookmaker', 'kind'])
    rejected = GaugeMetricFamily('prophitbet_scraper_rejected', 'Rejected observations in last import', labels=['bookmaker', 'reason'])
    for stage, filename, time_key in [('collection', 'snapshot.json', 'collected_at'), ('import', 'import-report.json', 'timestamp')]:
        try:
            path = root / filename
            if path.stat().st_size > 15_000_000:
                raise ValueError('oversized')
            data = json.loads(path.read_text(encoding='utf-8'))
            dt = datetime.fromisoformat(data[time_key])
            if dt.tzinfo is None or not isinstance(data.get('sources', {}), dict):
                raise ValueError('invalid telemetry')
            # Construct separately: malformed documents cannot leave partial metrics.
            entries = []
            for book in ('BETWAY', 'HOLLYWOODBETS'):
                source = data.get('sources', {}).get(book, {})
                state = source.get('status', data.get('status', 'unavailable'))
                state = state if state in STATUSES else 'other'
                count = len(source.get('events', [])) if stage == 'collection' else int(source.get('accepted', 0))
                reasons = {}
                for reason, value in source.get('rejected', {}).items():
                    label = reason if reason in REASONS else 'other'
                    reasons[label] = reasons.get(label, 0) + int(value)
                entries.append((book, state, count, reasons))
            elapsed = float(data.get('duration_seconds', 0))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            health.add_metric([stage], 0)
            continue
        health.add_metric([stage], 1)
        timestamp.add_metric([stage], dt.timestamp())
        duration.add_metric([stage], elapsed)
        if stage == 'import':
            overall = data.get('status', 'unavailable')
            status.add_metric([stage, 'ALL', overall if overall in STATUSES else 'other'], 1)
        for book, state, count, reasons in entries:
            status.add_metric([stage, book, state], 1)
            records.add_metric([stage, book, 'observed' if stage == 'collection' else 'accepted'], count)
            if stage == 'import':
                for reason in sorted(REASONS | {'other'}):
                    rejected.add_metric([book, reason], reasons.get(reason, 0))
    yield from (health, timestamp, duration, status, records, rejected)
