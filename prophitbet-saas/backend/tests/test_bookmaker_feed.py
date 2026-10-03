import unittest
from datetime import datetime, timedelta, timezone
from backend.app.services.bookmaker_feed import parse_betway


class BookmakerFeedTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 1, 10, tzinfo=timezone.utc)
        self.html = '''<div><a href="/event/soccer/test/a-b?eventId=123">
            <strong>A</strong><strong>B</strong>Today 15:30</a>
            <div><div price="x">2.1</div><div price="x">3.2</div><div price="x">3.4</div></div></div>'''
        self.source = dict(status='collected', observed_at=self.now.isoformat(),
            url='https://sports.betway.co.za/sport/soccer', headers=['1X2', '1 X 2'],
            browser_timezone='Africa/Johannesburg', clock='12:00:00', events=[self.html])

    def test_valid_and_deduplicated(self):
        self.source['events'] *= 2
        rows, rejected = parse_betway(self.source, self.now)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['match_date'].hour, 13)
        self.assertEqual(rows[0]['odds_draw'], 3.2)
        self.assertEqual(rejected, {})

    def test_stale_and_future(self):
        for delta in (-16, 1):
            self.source['observed_at'] = (self.now + timedelta(minutes=delta)).isoformat()
            with self.assertRaises(ValueError):
                parse_betway(self.source, self.now)

    def test_metadata_rejected(self):
        for key, value in [('clock', '08:00:00'), ('browser_timezone', 'UTC'),
                           ('headers', ['Totals']), ('url', 'https://example.com'),
                           ('observed_at', '2026-10-01T10:00:00')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_betway({**self.source, key: value}, self.now)

    def test_invalid_events(self):
        for old, new in [('Today', 'Saturday'), ('/test/', '/esoccer/'),
                         ('2.1', 'NaN'), ('price="x"', 'disabled price="x"'),
                         ('15:30', '09:30'), ('<strong>B</strong>', '<strong>A</strong>')]:
            with self.subTest(new=new):
                rows, errors = parse_betway({**self.source, 'events': [self.html.replace(old, new)]}, self.now)
                self.assertEqual(rows, [])
                self.assertTrue(errors)
