import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.app.workers import fixtures


class FixtureIntegrityTests(unittest.TestCase):
    def test_espn_stops_after_access_denial(self):
        async def run():
            client = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=403)))
            with patch.object(fixtures, "_espn_forbidden", False), \
                 patch.object(fixtures, "_get_async_client", AsyncMock(return_value=client)), \
                 patch.object(fixtures, "_request_semaphore", asyncio.Semaphore(1)):
                league = SimpleNamespace(country="Scotland", name="Premiership")
                self.assertEqual(await fixtures._scrape_espn_fixtures(league), [])
                self.assertEqual(await fixtures._scrape_espn_fixtures(league), [])
                self.assertEqual(client.get.call_count, 1)
        asyncio.run(run())

    def test_espn_ignores_bad_completed_and_unconfirmed_kickoffs(self):
        now = datetime(2026, 9, 19, tzinfo=timezone.utc)
        competition = dict(timeValid=True, competitors=[
            dict(homeAway="home", team=dict(displayName="Home")),
            dict(homeAway="away", team=dict(displayName="Away"))])
        event = dict(date="2026-09-20T14:00Z", status=dict(type=dict(name="STATUS_SCHEDULED")), competitions=[competition])
        events = [None, event, {**event, "date": "broken"}, {**event, "date": "2026-09-20T14:00"},
                  {**event, "competitions": [{**competition, "timeValid": False}]},
                  {**event, "status": dict(type=dict(name="STATUS_FULL_TIME"))},
                  {**event, "date": "2027-09-20T14:00Z"}]
        rows = fixtures._parse_espn_events(events, "https://example.com", now, now + timedelta(days=30))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["match_date"], datetime(2026, 9, 20, 14, tzinfo=timezone.utc))

    def test_espn_uses_months_across_year_boundary(self):
        async def run():
            response = SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                json=lambda: dict(leagues=[dict(slug="sco.1")], events=[]))
            client = SimpleNamespace(get=AsyncMock(return_value=response))
            with patch.object(fixtures, "datetime", wraps=datetime) as clock, \
                 patch.object(fixtures, "valid_fixture_rows", side_effect=lambda rows, now: rows), \
                 patch.object(fixtures, "_get_async_client", AsyncMock(return_value=client)), \
                 patch.object(fixtures, "_request_semaphore", asyncio.Semaphore(1)), \
                 patch.object(fixtures.asyncio, "sleep", AsyncMock()):
                clock.now.return_value = datetime(2026, 12, 20, tzinfo=timezone.utc)
                await fixtures._scrape_espn_fixtures(SimpleNamespace(country="Scotland", name="Premiership"))
                urls = [call.args[0] for call in client.get.call_args_list]
                self.assertEqual(len(urls), 2)
                self.assertIn("dates=202612&", urls[0])
                self.assertIn("dates=202701&", urls[1])
        asyncio.run(run())

    def test_espn_rejects_wrong_competition(self):
        async def run():
            response = SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                json=lambda: dict(leagues=[dict(slug="eng.1")], events=[]))
            client = SimpleNamespace(get=AsyncMock(return_value=response))
            with patch.object(fixtures, "_get_async_client", AsyncMock(return_value=client)), \
                 patch.object(fixtures, "_request_semaphore", asyncio.Semaphore(1)), \
                 patch.object(fixtures.asyncio, "sleep", AsyncMock()):
                self.assertEqual(await fixtures._scrape_espn_fixtures(SimpleNamespace(country="Scotland", name="Premiership")), [])
                self.assertEqual(client.get.call_count, 1)
        asyncio.run(run())

    def test_scraper_uses_public_feed_before_keyed_fallback(self):
        async def run():
            with patch.object(fixtures, "_fetch_football_data_fixtures", AsyncMock(return_value=[])), \
                 patch.object(fixtures, "_fetch_fixture_download_fixtures", AsyncMock(return_value=[])), \
                 patch.object(fixtures, "_scrape_espn_fixtures", AsyncMock(return_value=[{"source_url": "ESPN"}])), \
                 patch.object(fixtures, "_fetch_api_football_fixtures", AsyncMock()) as keyed:
                self.assertEqual(await fixtures._scrape_league_fixtures_async(object()), [{"source_url": "ESPN"}])
                keyed.assert_not_awaited()
        asyncio.run(run())

    def test_invalid_dates_and_duplicate_rows_are_rejected(self):
        now = datetime(2026, 9, 18, tzinfo=timezone.utc)
        row = dict(home_team="Home", away_team="Away", match_date=now + timedelta(days=1), source_url="https://example.com/feed")
        rows = [row, row.copy(), {**row, "match_date": now.replace(tzinfo=None)},
                {**row, "match_date": now - timedelta(seconds=1)},
                {**row, "source_url": None}, {**row, "away_team": "Home"}]
        self.assertEqual(fixtures.valid_fixture_rows(rows, now), [row])

    def test_offset_conversion_preserves_instant(self):
        now = datetime(2026, 9, 18, tzinfo=timezone.utc)
        date = datetime(2026, 9, 19, 0, 30, tzinfo=timezone(timedelta(hours=2)))
        rows = fixtures.valid_fixture_rows([dict(home_team="H", away_team="A", match_date=date, source_url="https://example.com")], now)
        self.assertEqual(rows[0]["match_date"].isoformat(), "2026-09-18T22:30:00+00:00")

    def test_completed_fixture_is_not_published_even_with_future_date(self):
        async def run():
            payload = [dict(HomeTeam="H", AwayTeam="A", DateUtc="2099-09-12 14:00:00Z", HomeTeamScore=2, AwayTeamScore=0)]
            response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
            client = SimpleNamespace(get=AsyncMock(return_value=response))
            with patch.object(fixtures, "_get_async_client", AsyncMock(return_value=client)), patch.object(fixtures, "_request_semaphore", asyncio.Semaphore(1)):
                self.assertEqual(await fixtures._fetch_fixture_download_fixtures(SimpleNamespace(country="England", name="Premier-League")), [])
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
