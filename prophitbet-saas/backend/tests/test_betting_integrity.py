import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from backend.app.services.betting_integrity import (
    valid_odds, normalize_selection, settlement, settle_legs, observed_quote,
)


class BettingIntegrityTests(unittest.TestCase):
    def test_invalid_prices(self):
        for price in (None, True, 0, 1, -2, float('nan'), float('inf'), '2.0'):
            self.assertFalse(valid_odds(price))
        self.assertTrue(valid_odds(2.1))

    def test_canonical_selections(self):
        self.assertEqual(normalize_selection('Home Win (1) (Hollywoodbets)'), 'H')
        self.assertEqual(normalize_selection('Over 2.5 Goals'), 'Over 2.5')
        with self.assertRaises(ValueError):
            normalize_selection('Any other market')

    def test_verified_regulation_results(self):
        for pick, actual, score, expected in (
            ('H', 'H', '2-1', 'WON'), ('A', 'H', '2-1', 'LOST'),
            ('Over 2.5', 'H', '2-1', 'WON'), ('Under 2.5', 'D', '1-1', 'WON'),
            ('BTTS Yes', 'D', '1-1', 'WON'), ('BTTS No', 'A', '0-1', 'WON'),
            ('1X', 'D', '0-0', 'WON'), ('12', 'D', '0-0', 'LOST'),
            ('X2', 'A', '0-1', 'WON'), ('DNB 1', 'D', '1-1', 'VOID'),
            ('DNB 2', 'A', '0-1', 'WON'), ('H', 'A', '2-1', None),
            ('H', 'H', None, None), ('unsupported', 'H', '2-1', None),
        ):
            with self.subTest(pick=pick, score=score):
                self.assertEqual(settlement(pick, actual, score), expected)

    def test_accumulator_void_and_pending_legs(self):
        won = {'status': 'WON', 'odds_taken': 2.5}
        void = {'status': 'VOID', 'odds_taken': 4.0}
        pending = {'status': 'PENDING', 'odds_taken': 3.0}
        lost = {'status': 'LOST', 'odds_taken': 2.0}
        self.assertEqual(settle_legs([won, void], 100), ('WON', 150.0))
        self.assertEqual(settle_legs([won, pending], 100), ('PENDING', 0.0))
        self.assertEqual(settle_legs([lost, pending], 100), ('LOST', -100.0))
        self.assertEqual(settle_legs([void, void], 100), ('VOID', 0.0))

    def test_quotes_require_provenance_and_freshness(self):
        now = datetime.now(timezone.utc)
        row = SimpleNamespace(bookmaker='BETWAY', source_url='https://www.betway.co.za/event/1',
            scraped_at=now, match_date=now + timedelta(days=1), home_team='Home', away_team='Away',
            markets_data={'provider_event_id': '1', 'provenance': 'observed', 'jurisdiction': 'ZA',
                          'period': 'regulation', 'suspended': False})
        fixture = SimpleNamespace(is_current=True, source_url='https://example.com/fixture',
            fetched_at=now, home_team='Home', away_team='Away', match_date=row.match_date)
        self.assertTrue(observed_quote(row, fixture, now))
        for field, value in (('source_url', 'https://betway.co.za.fake.test/1'),
                             ('scraped_at', now - timedelta(minutes=16)),
                             ('scraped_at', now + timedelta(seconds=1)),
                             ('match_date', now - timedelta(minutes=1)),
                             ('markets_data', {})):
            previous = getattr(row, field)
            setattr(row, field, value)
            self.assertFalse(observed_quote(row, fixture, now))
            setattr(row, field, previous)
        fixture.is_current = False
        self.assertFalse(observed_quote(row, fixture, now))


if __name__ == '__main__':
    unittest.main()
