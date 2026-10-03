"""Observed quotes and honest unavailable-feed contracts."""
import unittest
from unittest.mock import AsyncMock
from backend.app.services.sa_odds_service import sa_odds_service, dutch_stakes


class SAServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_invented_slips_or_database_writes(self):
        db = AsyncMock()
        result = await sa_odds_service.generate_perfect_bet_slips(db, bankroll_zar=1500)
        self.assertEqual(result['slips'], {})
        self.assertEqual(result['status'], 'manual_selection_required')
        db.execute.assert_not_called()
        db.commit.assert_not_called()

    async def test_missing_feed_is_not_reported_as_success(self):
        db = AsyncMock()
        result = await sa_odds_service.sync_all_sa_markets(db)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['records_synced'], 0)
        db.commit.assert_not_called()

    def test_unprofitable_prices_do_not_become_arbitrage(self):
        self.assertIsNone(dutch_stakes([1.99, 1.99], 100))
        self.assertIsNone(dutch_stakes([2, 2], 100))
        self.assertIsNone(dutch_stakes([float('nan'), 3], 100))
        self.assertIsNone(dutch_stakes([2.01, 2.01], .01))

    def test_cent_rounded_stakes(self):
        stakes, payout, profit = dutch_stakes([2.4, 3.6, 5.2], 100)
        self.assertAlmostEqual(sum(stakes), 100, places=2)
        self.assertGreater(profit, 0)
        self.assertGreater(payout, 100)
        for stake, price in zip(stakes, [2.4, 3.6, 5.2]):
            self.assertGreater(stake * price, 100)
