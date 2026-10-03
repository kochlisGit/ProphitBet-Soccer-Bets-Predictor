"""Read-only, cross-process metrics; no user data or job IDs are exported."""

import logging
import time
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler

import redis
from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from sqlalchemy import create_engine, text

from backend.app.config import get_settings
from backend.app.scraper_metrics import scraper_metrics

logger = logging.getLogger(__name__)


def gauge(name, help_text, value):
    return GaugeMetricFamily(name, help_text, value=float(value))


class DomainCollector:
    """Refresh off the HTTP thread; failed dependencies omit data, not fake zeros."""

    def __init__(self, client, engine):
        self.redis = client
        self.engine = engine
        self.snapshot = ()
        self.quality_snapshot = ()
        self.quality_next_check = 0
        self.quality_signature = None

    def describe(self):
        return []

    def collect(self):
        yield from self.snapshot

    def redis_metrics(self):
        depth = GaugeMetricFamily('prophitbet_queue_depth', 'Waiting broker messages, including priority queues', labels=['queue'])
        for queue in ('celery', 'data_sync'):
            keys = [queue] + [f'{queue}\x06\x16{p}' for p in (3, 6, 9)]
            with self.redis.pipeline(transaction=False) as pipe:
                for key in keys:
                    pipe.llen(key)
                depth.add_metric([queue], sum(pipe.execute()))
        yield depth
        yield gauge('prophitbet_broker_unacked', 'Reserved or executing broker messages (not just running tasks)', self.redis.hlen('unacked'))
        oldest = self.redis.zrange('unacked_index', 0, 0, withscores=True)
        yield gauge('prophitbet_broker_oldest_unacked_seconds', 'Age since oldest unacknowledged delivery', max(0, time.time() - oldest[0][1]) if oldest else 0)
        shared = self.redis.hgetall('prophitbet:monitoring:metrics')
        tasks = CounterMetricFamily('prophitbet_shared_tasks', 'Recorded task completions across all workers since Redis reset', labels=['status'])
        for status, key in [('success', 'successful_tasks'), ('failure', 'failed_tasks'), ('retry', 'retried_tasks')]:
            tasks.add_metric([status], float(shared.get(key, 0)))
        yield tasks

    def database_metrics(self):
        with self.engine.connect() as connection:
            connection.execute(text('SET TRANSACTION READ ONLY'))
            # Fixed allowlist only; no labels containing people, teams, URLs or IDs.
            counts = GaugeMetricFamily('prophitbet_records', 'Current stored records, not lifetime event counters', labels=['entity'])
            for table in ('predictions', 'fixtures', 'trained_models', 'league_datasets', 'sa_bookmaker_odds'):
                counts.add_metric([table], connection.scalar(text(f'SELECT count(*) FROM {table}')))
            yield counts
            sync = GaugeMetricFamily('prophitbet_league_sync_age_seconds', 'Seconds since each league last synced; absent if never synced', labels=['league_id'])
            rows = connection.execute(text('SELECT id, extract(epoch FROM (now() - last_synced_at)) FROM leagues WHERE is_active AND last_synced_at IS NOT NULL'))
            for league_id, age in rows:
                sync.add_metric([str(league_id)], max(0, float(age)))
            yield sync
            yield gauge('prophitbet_leagues_never_synced', 'Active leagues without a successful sync timestamp', connection.scalar(text('SELECT count(*) FROM leagues WHERE is_active AND last_synced_at IS NULL')))
            yield gauge('prophitbet_upcoming_fixtures', 'Fresh sourced current fixtures in the next seven days', connection.scalar(text("SELECT count(*) FROM fixtures WHERE is_current AND source_url IS NOT NULL AND fetched_at BETWEEN now()-interval '48 hours' AND now() AND match_date BETWEEN now() AND now() + interval '7 days'")))
            yield gauge('prophitbet_recent_predictions', 'Prediction records created or refreshed in the last 24 hours', connection.scalar(text("SELECT count(*) FROM predictions WHERE created_at >= now() - interval '24 hours'")))
            yield gauge('prophitbet_recent_odds_records', 'Odds records scraped in the last 24 hours; not a verification claim', connection.scalar(text("SELECT count(*) FROM sa_bookmaker_odds WHERE scraped_at >= now() - interval '24 hours'")))

    def refresh(self):
        metrics = []
        health = GaugeMetricFamily('prophitbet_metrics_source_up', 'Whether this refresh could read a metrics source', labels=['source'])
        for source, fetch in [('redis', self.redis_metrics), ('postgres', self.database_metrics)]:
            try:
                result = list(fetch())
            except Exception:
                logger.exception('Metrics source unavailable: %s', source)
                health.add_metric([source], 0)
            else:
                metrics.extend(result)
                health.add_metric([source], 1)
        metrics.extend([health, gauge('prophitbet_metrics_refresh_timestamp_seconds', 'Last completed collection attempt', time.time())])
        metrics.extend(scraper_metrics())
        if time.time() >= self.quality_next_check:
            self.quality_next_check = time.time() + 300
            try:
                from backend.app.quality_metrics import quality_metrics
                quality = quality_metrics(self.engine)
                self.quality_snapshot = tuple(quality) + (gauge('prophitbet_quality_check_up', 'Last integrity and storage audit completed', 1),
                    gauge('prophitbet_quality_check_timestamp_seconds', 'Latest completed integrity audit', time.time()))
                summary = {m.name: [{**s.labels, 'value': s.value} for s in m.samples] for m in quality}
                signature = json.dumps(summary, sort_keys=True)
                if signature != self.quality_signature:
                    from datetime import datetime, timezone
                    folder = Path('/app/storage/bookmaker-feed/logs')
                    folder.mkdir(parents=True, exist_ok=True)
                    handler = RotatingFileHandler(folder / 'validation.jsonl', maxBytes=2_000_000, backupCount=3)
                    try:
                        entry = dict(timestamp=datetime.now(timezone.utc).isoformat(), stage='validation', checks=summary)
                        handler.emit(logging.LogRecord('validation', logging.INFO, '', 0, json.dumps(entry), (), None))
                    finally:
                        handler.close()
                    self.quality_signature = signature
            except Exception:
                logger.exception('Data integrity audit unavailable')
                self.quality_snapshot = (gauge('prophitbet_quality_check_up', 'Last integrity and storage audit completed', 0),)
        metrics.extend(self.quality_snapshot)
        self.snapshot = tuple(metrics)


def main():
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=3, socket_connect_timeout=3)
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True, pool_size=1, max_overflow=0, connect_args={'connect_timeout': 3, 'options': '-c statement_timeout=3000'})
    collector = DomainCollector(client, engine)
    collector.refresh()
    REGISTRY.register(collector)
    start_http_server(9108)
    while True:
        time.sleep(30)
        collector.refresh()


if __name__ == '__main__':
    main()
