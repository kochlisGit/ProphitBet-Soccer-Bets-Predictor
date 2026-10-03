import json
from pathlib import Path
import tempfile
import unittest
from prometheus_client import CollectorRegistry, generate_latest
from backend.app.scraper_metrics import scraper_metrics


class ScraperMetricsTests(unittest.TestCase):
    def render(self, root):
        class Collector:
            def collect(self):
                return scraper_metrics(root)
        registry = CollectorRegistry()
        registry.register(Collector())
        return generate_latest(registry).decode()

    def test_missing_is_unavailable_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.render(Path(directory))
        self.assertIn('prophitbet_scraper_telemetry_available{stage="collection"} 0.0', output)
        self.assertNotIn('prophitbet_scraper_timestamp_seconds{', output)

    def test_real_counts_and_bounded_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'snapshot.json').write_text(json.dumps(dict(collected_at='2026-10-01T18:00:00+00:00',
                sources={'BETWAY': dict(status='collected', events=['raw-secret-html'])})))
            (root / 'import-report.json').write_text(json.dumps(dict(timestamp='2026-10-01T18:01:00+00:00',
                sources={'BETWAY': dict(status='collected', accepted=2, rejected={'private-url': 3})})))
            output = self.render(root)
        self.assertIn('bookmaker="BETWAY",kind="accepted",stage="import"} 2.0', output)
        self.assertIn('bookmaker="BETWAY",reason="other"} 3.0', output)
        self.assertNotIn('private-url', output)
        self.assertNotIn('raw-secret-html', output)

    def test_invalid_json_and_naive_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'snapshot.json').write_text('{')
            (root / 'import-report.json').write_text('{"timestamp":"2026-10-01T18:00:00"}')
            output = self.render(root)
        self.assertIn('prophitbet_scraper_telemetry_available{stage="import"} 0.0', output)
        self.assertNotIn('prophitbet_scraper_records{', output)
