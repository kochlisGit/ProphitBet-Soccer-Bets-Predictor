"""Exercise publication predicates against real SQL, without a production DB."""
import unittest
from datetime import datetime, timedelta, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, MetaData, String, Table, create_engine, func, select

from backend.app.db.models import Prediction
from backend.app.services.data_integrity import publishable_prediction


class PredictionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 19, tzinfo=timezone.utc)
        self.engine = create_engine("sqlite://")
        metadata = MetaData()
        common = lambda: [Column("league_id", Integer), Column("home_team", String),
                          Column("away_team", String), Column("match_date", DateTime)]
        self.predictions = Table("predictions", metadata, *common(),
            Column("actual_result", String), Column("actual_score", String),
            Column("result_source", String), Column("result_verified_at", DateTime))
        self.fixtures = Table("fixtures", metadata, Column("id", Integer), *common(),
            Column("is_current", Boolean), Column("source_url", String), Column("fetched_at", DateTime))
        metadata.create_all(self.engine)
        self.match = dict(league_id=6, home_team="Home", away_team="Away", match_date=self.now + timedelta(days=1))

    def tearDown(self):
        self.engine.dispose()

    def count(self, prediction=None, fixture=None):
        with self.engine.begin() as connection:
            connection.execute(self.predictions.insert(), prediction or self.match)
            if fixture:
                connection.execute(self.fixtures.insert(), fixture)
            return connection.scalar(select(func.count()).select_from(Prediction).where(publishable_prediction(self.now)))

    def fixture(self, **overrides):
        return dict(self.match, id=1, is_current=True, source_url="https://example.com", fetched_at=self.now) | overrides

    def test_exact_fresh_fixture_is_publishable(self):
        self.assertEqual(self.count(fixture=self.fixture()), 1)

    def test_unsourced_legacy_prediction_is_hidden(self):
        self.assertEqual(self.count(), 0)

    def test_rescheduled_fixture_does_not_validate_old_prediction(self):
        self.assertEqual(self.count(fixture=self.fixture(match_date=self.now + timedelta(days=2))), 0)

    def test_retired_fixture_is_hidden(self):
        self.assertEqual(self.count(fixture=self.fixture(is_current=False)), 0)

    def test_expired_snapshot_is_hidden(self):
        self.assertEqual(self.count(fixture=self.fixture(fetched_at=self.now - timedelta(hours=49))), 0)

    def test_verified_history_survives_fixture_retirement(self):
        prediction = self.match | dict(match_date=self.now - timedelta(days=1), actual_result="H",
            actual_score="2-0", result_source="https://example.com", result_verified_at=self.now)
        self.assertEqual(self.count(prediction=prediction), 1)

    def test_unverified_score_is_hidden(self):
        prediction = self.match | dict(match_date=self.now - timedelta(days=1), actual_result="H", actual_score="2-0")
        self.assertEqual(self.count(prediction=prediction), 0)


if __name__ == "__main__":
    unittest.main()
