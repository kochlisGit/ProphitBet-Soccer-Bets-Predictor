import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
from sqlalchemy import select

from backend.app.api.sa_markets import AISlipRequest
from backend.app.api.betslip import BetLegRequest
from backend.app.db.models import Prediction, TrainedModel
from backend.app.services.ai_slip_builder import rank_slip
from backend.app.services.prediction_markets import market_selection, settle_prediction_selection
from backend.app.workers.results import _find_match_result
from backend.tests.test_ai_slip_builder import row, quote


class MultiMarketTests(unittest.TestCase):
    def test_market_comes_from_correct_model(self):
        sql = str(select(Prediction).where(Prediction.market_type == 'btts'))
        self.assertIn('trained_models.id = predictions.model_id', sql)
        self.assertIn('trained_models.target_type', sql)

    def build(self, rows, quotes=None, **kwargs):
        options = dict(bookmaker='BETWAY', legs=1, min_probability=.5, strategy='confidence',
                       draft=True, min_ev=3, market_type='all')
        options.update(kwargs)
        return rank_slip(rows, quotes or {}, **options)

    def test_corner_draft_is_named_and_unpriced(self):
        pred, fixture = row()
        pred.market_type, pred.probabilities = 'corners-9.5', {'Under': .2, 'Over': .8}
        result = self.build([(pred, fixture)])
        self.assertEqual(result['legs'][0]['selection'], 'Corners Over 9.5')
        self.assertIsNone(result['legs'][0]['odds_taken'])
        self.assertEqual(self.build([(pred, fixture)], draft=False)['legs'], [])
        self.assertEqual(self.build([(pred, fixture)], market_type='btts')['legs'], [])

    def test_goal_quote_uses_correct_market_price(self):
        pred, fixture = row()
        pred.market_type, pred.probabilities = 'over-under', {'Under': .2, 'Over': .8}
        prices = quote(fixture, price=9)
        prices[str(fixture.id)]['betway']['over_25'] = 1.7
        result = self.build([(pred, fixture)], prices, draft=False)
        self.assertEqual(result['legs'][0]['odds_taken'], 1.7)
        self.assertEqual(result['legs'][0]['selection'], 'Over 2.5')

    def test_no_same_match_accumulator_across_markets(self):
        pred, fixture = row()
        other = SimpleNamespace(**vars(pred))
        other.market_type, other.probabilities = 'btts', {'No': .2, 'Yes': .8}
        self.assertEqual(len(self.build([(pred, fixture), (other, fixture)], legs=2)['legs']), 1)

    def test_factual_market_settlement(self):
        df = pd.DataFrame([dict(Date='2026-09-01', Home='Home', Away='Away', HG=2, AG=1,
                                Result='H', HC=6, AC=4, HST=4, AST=3)])
        date = datetime(2026, 9, 1, tzinfo=timezone.utc)
        for market, expected in [('result', 'H'), ('over-under', 'Over'), ('btts', 'Yes'),
                                 ('corners-9.5', 'Over'), ('shots-target-7.5', 'Under')]:
            result = _find_match_result(df, 'Home', 'Away', date, market)
            self.assertEqual(result['result'], expected)
        self.assertIsNone(_find_match_result(df.drop(columns=['HC']), 'Home', 'Away', date, 'corners-9.5'))

    def test_journal_wont_settle_corners_from_goals(self):
        pred = SimpleNamespace(market_type='result', actual_result='H', actual_score='2-1')
        self.assertIsNone(settle_prediction_selection('Corners Over 9.5', pred))
        pred.market_type, pred.actual_result = 'corners-9.5', 'Over'
        self.assertEqual(settle_prediction_selection('Corners Over 9.5', pred), 'WON')
        self.assertEqual(settle_prediction_selection('Corners Under 9.5', pred), 'LOST')
        self.assertIsNone(settle_prediction_selection('Corners Over 10.5', pred))

    def test_request_accepts_markets_and_explicit_selections(self):
        self.assertEqual(AISlipRequest(bookmaker='BETWAY', market_type='btts').market_type, 'btts')
        with self.assertRaises(ValueError):
            AISlipRequest(bookmaker='BETWAY', market_type='imaginary')
        leg = BetLegRequest(bookmaker='BETWAY', match_title='Home vs Away', odds_taken=2,
                            selection=market_selection('shots-target-8.5', 'Under'))
        self.assertEqual(leg.selection, 'Shots on target Under 8.5')
