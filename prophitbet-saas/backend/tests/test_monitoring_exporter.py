import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from backend.app.middleware import RateLimitMiddleware
from backend.app.monitoring_exporter import DomainCollector


class MonitoringTests(unittest.TestCase):
    def test_metrics_bypass_rate_limit_but_api_does_not(self):
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.get('/metrics')
        @app.get('/normal')
        def endpoint():
            return {'ok': True}

        settings = MagicMock(RATE_LIMIT_REQUESTS=1, RATE_LIMIT_WINDOW_SECONDS=60)
        from unittest.mock import AsyncMock
        client = MagicMock()
        client.incr = AsyncMock(return_value=2)
        client.aclose = AsyncMock()
        with patch('backend.app.middleware.get_settings', return_value=settings), patch('backend.app.middleware.Redis.from_url', return_value=client):
            with TestClient(app) as http:
                self.assertEqual(http.get('/metrics').status_code, 200)
                client.incr.assert_not_called()
                self.assertEqual(http.get('/normal').status_code, 429)

    def test_shared_metrics_and_priority_queues(self):
        client = MagicMock()
        pipe = client.pipeline.return_value.__enter__.return_value
        pipe.execute.return_value = [2, 1, 0, 0]
        client.hlen.return_value = 4
        client.zrange.return_value = []
        client.hgetall.return_value = {'successful_tasks': '12', 'failed_tasks': '3'}
        collector = DomainCollector(client, MagicMock())
        collector.snapshot = tuple(collector.redis_metrics())
        registry = CollectorRegistry()
        registry.register(collector)
        output = generate_latest(registry).decode()
        self.assertIn('prophitbet_queue_depth{queue="data_sync"} 3.0', output)
        self.assertIn('prophitbet_shared_tasks_total{status="success"} 12.0', output)

    def test_failed_source_omits_data_and_sets_health_zero(self):
        collector = DomainCollector(MagicMock(), MagicMock())
        with patch.object(collector, 'redis_metrics', side_effect=RuntimeError('offline')), patch.object(collector, 'database_metrics', return_value=[]), self.assertLogs('backend.app.monitoring_exporter', level='ERROR'):
            collector.refresh()
        registry = CollectorRegistry()
        registry.register(collector)
        output = generate_latest(registry).decode()
        self.assertIn('prophitbet_metrics_source_up{source="redis"} 0.0', output)
        self.assertNotIn('prophitbet_queue_depth', output)


if __name__ == '__main__':
    unittest.main()
