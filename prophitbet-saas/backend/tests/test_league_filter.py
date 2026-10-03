import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException
from backend.app.services.league_filter import resolve_league_id


class LeagueFilterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = AsyncMock()
        result = MagicMock()
        result.all.return_value = [SimpleNamespace(id=606, country='England', name='Premier-League'),
            SimpleNamespace(id=2626, country='Russia', name='Premier-League'),
            SimpleNamespace(id=27, country='Scotland', name='Premiership')]
        self.db.execute.return_value = result

    async def test_epl_uses_country_not_hardcoded_id(self):
        self.assertEqual(await resolve_league_id(self.db, 'Premier League'), 606)
        self.assertEqual(await resolve_league_id(self.db, 'Russian Premier League'), 2626)

    async def test_explicit_id(self):
        self.assertEqual(await resolve_league_id(self.db, 'league:2626'), 2626)
        self.db.execute.assert_not_called()

    async def test_scotland_not_south_africa(self):
        self.assertEqual(await resolve_league_id(self.db, 'Premiership'), 27)

    async def test_substring_and_wildcards_rejected(self):
        for query in ['Premier', '%', 'Unknown League', 'league:bad']:
            with self.assertRaises(HTTPException):
                await resolve_league_id(self.db, query)
