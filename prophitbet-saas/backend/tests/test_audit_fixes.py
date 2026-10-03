import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.api.betslip import selection_won
from backend.app.api.models import AutoTuneRequest, _user_plan, _reserve_training, _dispatch_training
from backend.app.services import model_service, prediction_service
from backend.app.workers.results import _find_match_result
from src.network.leagues.downloaders.public_json import parse_completed_matches


class DataTests(unittest.TestCase):
    def test_prediction_generation_does_not_publish_externally_by_default(self):
        import inspect
        from backend.app.workers.predict import generate_daily_predictions_task
        self.assertIs(inspect.signature(generate_daily_predictions_task.run).parameters["publish_external"].default, False)

    def test_safe_artifact_target_round_trip_and_legacy_compatibility(self):
        import skops.io as sio
        from src.models.classifiers import RandomForest
        from src.preprocessing.utils.target import TargetType
        from backend.app.services.model_artifact import serialize_model, deserialize_model
        model = RandomForest(league_id="test", model_id="test", target_type=TargetType.RESULT,
                             calibrate_probabilities=False)
        for artifact in (serialize_model(model), sio.dumps(model)):
            restored = deserialize_model(artifact)
            self.assertIs(restored._target_type, TargetType.RESULT)
        self.assertIs(model._target_type, TargetType.RESULT)

    def test_cup_scores_preserve_scope_and_reject_bad_data(self):
        match = dict(HomeTeam="Home", AwayTeam="Away", DateUtc="2025-01-01 20:00:00Z",
                     HomeTeamScore=2, AwayTeamScore=1, RoundNumber=1)
        rows = parse_completed_matches([match, match | {"RoundNumber": 10},
            match | {"HomeTeamScore": None}, match | {"HomeTeamScore": -1},
            match | {"HomeTeamScore": 1.5}, match | {"DateUtc": "2099-01-01T00:00:00Z"}],
            2024, "https://example.com")
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["ScoreScope"] for r in rows], ["regulation", "provider_final"])
        self.assertEqual(rows[0]["Result"], "H")
        self.assertIsNone(_find_match_result(pd.DataFrame([rows[1]]), "Home", "Away",
            datetime(2025, 1, 1, tzinfo=timezone.utc)))

    def test_total_markets_and_unknowns(self):
        self.assertTrue(selection_won("Over 2.5", "H", "2-1"))
        self.assertFalse(selection_won("Under 2.5", "H", "2-1"))
        self.assertTrue(selection_won("Under", "D", "0-0"))
        self.assertTrue(selection_won("H", "H", "2-0"))
        self.assertIsNone(selection_won("Over", "H", None))
        self.assertIsNone(selection_won("BTTS", "H", "2-1"))

    def test_metrics_use_only_evaluation_rows(self):
        fake = Mock()
        with patch.object(model_service, "get_model_class", return_value=Mock(return_value=fake)), \
             patch.object(model_service, "serialize_model", return_value=b"safe"), \
             patch("src.models.temporal_evaluation.evaluate_temporally", return_value={'folds': [
                 {'model': {'accuracy': .4, 'precision': .3, 'recall': .2, 'f1': .2, 'log_loss': 1., 'brier': .6},
                  'baseline': {'log_loss': 1.1}}]}):
            frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=60),
                'Home': ['Home'] * 60, 'Away': ['Away'] * 60, 'HG': [1, 0, 0] * 20,
                'AG': [0, 0, 1] * 20, 'feature': range(60), 'Result': ['H', 'D', 'A'] * 20})
            result = model_service.train_model(frame, "random_forest", use_odds=False)
        self.assertEqual(result["metrics"]["Accuracy"], .4)
        self.assertFalse(fake._requires_odds)

    def test_odds_free_predictions_and_home_away_class_order(self):
        model = SimpleNamespace(target_type="result", _requires_odds=False, _input_columns=["Date", "Home", "Away"],
            classifier=SimpleNamespace(classes_=np.array([0, 1, 2])),
            predict=lambda df: (np.array([0]), None),
            predict_proba=lambda df: np.array([[.7, .2, .1]]))
        df = pd.DataFrame([dict(Date="2025-01-01", Home="Home", Away="Away")])
        with patch.object(prediction_service, "deserialize_model", return_value=model), \
             patch("src.preprocessing.utils.inputs.construct_inputs_by_teams", return_value=df) as inputs:
            result = prediction_service.generate_predictions_for_league(b"safe", df,
                [dict(home_team="Home", away_team="Away", fixture_id="fixture-1")])
        self.assertEqual(result[0]["predicted_result"], "H")
        self.assertEqual(result[0]["probabilities"]["H"], .7)
        self.assertEqual(result[0]["fixture_id"], "fixture-1")
        self.assertFalse(inputs.call_args.kwargs["require_odds"])

    def test_trial_bounds_and_admin_plan(self):
        for trials in (0, 101, 1000000000):
            with self.assertRaises(ValidationError):
                AutoTuneRequest(league_id=1, model_type="logistic", n_trials=trials)
        self.assertEqual(_user_plan(SimpleNamespace(is_admin=True, subscription=None)), "elite")


class QuotaTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_jobs_consume_quota(self):
        user = SimpleNamespace(id=uuid4(), is_admin=False,
            subscription=SimpleNamespace(status="active", plan="pro"))
        db = Mock(commit=AsyncMock(), execute=AsyncMock(side_effect=[Mock(), Mock(scalar=lambda: 1), Mock(scalar=lambda: 2)]))
        with self.assertRaises(HTTPException) as raised:
            await _reserve_training(db, user)
        self.assertEqual(raised.exception.status_code, 429)
        db.add.assert_not_called()

    async def test_queue_failure_releases_reservation(self):
        reservation = SimpleNamespace(id=uuid4(), status="queued")
        db = Mock(commit=AsyncMock())
        with patch("backend.app.api.models._reserve_training", AsyncMock(return_value=reservation)):
            with self.assertRaises(HTTPException):
                await _dispatch_training(db, Mock(), Mock(apply_async=Mock(side_effect=RuntimeError())))
        self.assertEqual(reservation.status, "failed")
        db.commit.assert_awaited_once()

    async def test_subscription_recovers_plan_from_price(self):
        from backend.app.api import billing
        sub = SimpleNamespace(plan="free", status="past_due")
        db = Mock(execute=AsyncMock(return_value=Mock(scalar_one_or_none=lambda: sub)))
        request = Mock(body=AsyncMock(return_value=b"event"), headers={})
        event = {"type": "customer.subscription.updated", "data": {"object": {
            "id": "sub_test", "status": "active", "items": {"data": [{"price": {"id": "price_test"}}]}}}}
        with patch("stripe.Webhook.construct_event", return_value=event), \
             patch.object(billing.settings, "STRIPE_PRICE_PRO", "price_test"):
            await billing.stripe_webhook(request, db)
            self.assertEqual((sub.plan, sub.status), ("pro", "active"))
            event["data"]["object"]["status"] = "past_due"
            await billing.stripe_webhook(request, db)
            self.assertEqual((sub.plan, sub.status), ("pro", "past_due"))


if __name__ == "__main__":
    unittest.main()
