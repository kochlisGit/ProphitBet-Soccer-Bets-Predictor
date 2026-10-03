import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
from pydantic import ValidationError
from fastapi import HTTPException
from backend.app.api.betslip import CreateBetSlipRequest, UpdateBetSlipRequest, create_bet_slip


def leg(title='Home vs Away', book='BETWAY', **kwargs):
    return dict(match_title=title, bookmaker=book, selection='H', odds_taken=2.5, **kwargs)


class RequestTests(unittest.TestCase):
    def test_invalid_numbers_and_statuses(self):
        for value in (float('nan'), float('inf'), -1, 0):
            with self.assertRaises(ValidationError):
                CreateBetSlipRequest(odds_taken=2, stake_amount=value)
        with self.assertRaises(ValidationError):
            UpdateBetSlipRequest(status='FAKE')
        with self.assertRaises(ValidationError):
            UpdateBetSlipRequest(pnl=1000)

    def test_one_book_and_distinct_events(self):
        with self.assertRaises(ValidationError):
            CreateBetSlipRequest(legs=[leg(), leg('Other vs Team', 'HOLLYWOODBETS')])
        with self.assertRaises(ValidationError):
            CreateBetSlipRequest(legs=[leg(), leg()])


class PersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_ai_prediction_link_requires_exact_fixture(self):
        now = datetime.now(timezone.utc)
        fixture = SimpleNamespace(id=uuid4(), league_id=1, home_team='Home', away_team='Away',
            match_date=now+timedelta(days=1), is_current=True, fetched_at=now,
            source_url='https://example.com/fixture')
        quote = SimpleNamespace(id=uuid4(), fixture_id=fixture.id, prediction_id=None,
            bookmaker='BETWAY', match_title='Home vs Away', home_team='Home', away_team='Away',
            match_date=fixture.match_date, scraped_at=now, odds_home=2.5,
            source_url='https://www.betway.co.za/event/1',
            markets_data={'provider_event_id':'1','provenance':'observed','jurisdiction':'ZA',
                          'period':'regulation','suspended':False})
        for league_id in (1, 2):
            prediction = SimpleNamespace(id=uuid4(), league_id=league_id, home_team='Home',
                away_team='Away', match_date=fixture.match_date)
            db = AsyncMock()
            db.add = Mock()
            db.get.side_effect = [quote, fixture, prediction]
            payload = CreateBetSlipRequest(legs=[leg(quote_id=quote.id, prediction_id=prediction.id)], stake_amount=100)
            await create_bet_slip(payload, db, SimpleNamespace(id=uuid4()))
            saved = db.add.call_args.args[0]
            self.assertEqual(saved.legs[0]['prediction_id'], str(prediction.id) if league_id == 1 else None)

    async def test_accumulator_saves_one_record_with_one_stake(self):
        db = AsyncMock()
        db.add = Mock()
        payload = CreateBetSlipRequest(legs=[leg(), leg('Other vs Team')], stake_amount=100)
        await create_bet_slip(payload, db, SimpleNamespace(id=uuid4()))
        db.add.assert_called_once()
        saved = db.add.call_args.args[0]
        self.assertEqual(saved.odds_taken, 6.25)
        self.assertEqual(saved.stake_amount, 100)
        self.assertEqual(len(saved.legs), 2)
        self.assertEqual(saved.selection, 'ACCUMULATOR')
        self.assertEqual(saved.odds_origin, 'manual_unverified')
        db.commit.assert_awaited_once()

    async def test_missing_quote_rejected_without_writes(self):
        db = AsyncMock()
        db.add = Mock()
        db.get.return_value = None
        payload = CreateBetSlipRequest(legs=[leg(quote_id=uuid4())], stake_amount=100)
        with self.assertRaises(HTTPException) as caught:
            await create_bet_slip(payload, db, SimpleNamespace(id=uuid4()))
        self.assertEqual(caught.exception.status_code, 409)
        db.add.assert_not_called()
        db.commit.assert_not_called()
