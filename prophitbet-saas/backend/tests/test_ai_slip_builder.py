import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as Obj
from uuid import uuid4
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError
from backend.app.api.sa_markets import AISlipRequest
from backend.app.services.ai_slip_builder import rank_slip, generate_ai_slip


def row(home='Home', away='Away', probability=.7):
    now = datetime.now(timezone.utc)
    fixture = Obj(id=uuid4(), league_id=1, home_team=home, away_team=away,
        match_date=now+timedelta(days=1), source_url='https://example.com/fixture')
    prediction = Obj(id=uuid4(), model_id=uuid4(), created_at=now,
        probabilities={'H': probability, 'D': .2, 'A': 1-probability-.2})
    return prediction, fixture


def build(rows, quotes=None, **kwargs):
    options = dict(bookmaker='BETWAY', legs=2, min_probability=.5,
                   strategy='confidence', draft=False, min_ev=3)
    options.update(kwargs)
    return rank_slip(rows, quotes or {}, **options)


def quote(fixture, book='betway', price=2):
    return {str(fixture.id): {book: dict(odds_home=price, odds_draw=3, odds_away=4,
        quote_id=str(uuid4()), source_url='https://www.betway.co.za/event',
        scraped_at=datetime.now(timezone.utc).isoformat())}}


class BuilderTests(unittest.TestCase):
    def test_missing_prices_are_not_filled(self):
        result = build([row()])
        self.assertEqual(result['status'], 'insufficient_data')
        self.assertEqual(result['legs'], [])
        self.assertIsNone(result['combined_odds'])
        self.assertFalse(result['can_save'])

    def test_draft_is_unpriced_and_not_saveable(self):
        result = build([row(), row('Other', 'Team')], draft=True)
        self.assertEqual(result['status'], 'draft')
        self.assertIsNone(result['combined_odds'])
        self.assertFalse(result['can_save'])
        self.assertTrue(all(p['odds_taken'] is None and p['quote_id'] is None for p in result['legs']))
        self.assertAlmostEqual(result['estimated_win_probability'], .49)

    def test_one_bookmaker_and_actual_product(self):
        rows = [row(), row('Other', 'Team')]
        prices = {**quote(rows[0][1], price=2), **quote(rows[1][1], price=1.8)}
        result = build(rows, prices)
        self.assertEqual(result['status'], 'ready')
        self.assertAlmostEqual(result['combined_odds'], 3.6)
        self.assertTrue(result['can_save'])
        self.assertEqual(build(rows, prices, bookmaker='HOLLYWOODBETS')['legs'], [])

    def test_duplicate_events_and_repeated_teams_excluded(self):
        first = row()
        result = build([first, first, row('Home', 'Third')], draft=True)
        self.assertEqual(len(result['legs']), 1)
        self.assertEqual(result['status'], 'insufficient_data')

    def test_reject_malformed_probabilities(self):
        for probabilities in ({'H': float('nan'), 'D': .2, 'A': .1}, {'H': True, 'D': 0, 'A': 0},
                              {'H': .9, 'D': .9, 'A': .9}, [1, 2, 3]):
            pred, fixture = row()
            pred.probabilities = probabilities
            self.assertEqual(build([(pred, fixture)], draft=True)['legs'], [])

    def test_no_invented_value_or_threshold_relaxation(self):
        rows = [row(probability=.6)]
        self.assertEqual(build(rows, quote(rows[0][1], price=1.2), strategy='value')['legs'], [])
        self.assertEqual(build(rows, draft=True, min_probability=.8)['legs'], [])

    def test_latest_prediction_not_cherry_picked(self):
        pred, fixture = row(probability=.4)
        older, _ = row(probability=.75)
        self.assertEqual(build([(pred, fixture), (older, fixture)], draft=True)['legs'], [])

    def test_request_bounds(self):
        for fields in ({'legs': 0}, {'legs': 7}, {'min_probability': float('nan')},
                       {'min_probability': 1.1}, {'strategy': 'guaranteed'}):
            with self.assertRaises(ValidationError):
                AISlipRequest(bookmaker='BETWAY', **fields)


class BuilderServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_draft_is_read_only_and_never_syncs_or_fabricates_quotes(self):
        db = AsyncMock()
        db.execute.return_value = Mock(all=Mock(return_value=[row()]))
        with patch('backend.app.services.ai_slip_builder.sa_odds_service.get_paired_odds_comparison', new_callable=AsyncMock) as prices:
            result = await generate_ai_slip(db, **AISlipRequest(bookmaker='BETWAY', draft=True, legs=1).model_dump())
            self.assertEqual(result['status'], 'draft')
            prices.assert_not_called()
        db.commit.assert_not_called()
        statement = str(db.execute.call_args.args[0])
        self.assertIn('is_house_model IS true', statement)
        self.assertIn('fetched_at BETWEEN', statement)
        self.assertIn('predictions.created_at BETWEEN', statement)
