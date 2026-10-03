import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.app.services.public_scraper import PublicScraper, ScrapeError, parse_page, validate_url


class ParserTests(unittest.TestCase):
    def test_html_records_keep_missing_values_and_provenance(self):
        result = parse_page('https://example.com', 'text/html',
                            b'<div class="event"><b>Home</b><a href="/match">View</a></div>',
                            rows='.event', fields={'home': 'b', 'away': '.away', 'link': 'a::href'})
        self.assertEqual(result['records'], [{'home': 'Home', 'away': None, 'link': '/match'}])
        self.assertFalse(result['verified'])
        self.assertEqual(len(result['sha256']), 64)

    def test_text_and_links(self):
        result = parse_page('https://example.com', 'text/html',
                            b'<title>Hi</title><script>secret</script><a href="/next">Next</a>')
        self.assertNotIn('secret', result['text'])
        self.assertEqual(result['links'][0]['url'], 'https://example.com/next')

    def test_json(self):
        self.assertEqual(parse_page('https://example.com', 'application/json', b'{"odds":null}')['data'], {'odds': None})

    def test_invalid_and_unsupported(self):
        for media, body in [('application/json', b'bad'), ('image/png', b'png')]:
            with self.assertRaises(ScrapeError):
                parse_page('https://example.com', media, body)


class RequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_address_rejected(self):
        with patch('socket.getaddrinfo', return_value=[(2, 1, 6, '', ('127.0.0.1', 80))]):
            with self.assertRaises(ScrapeError):
                await validate_url('http://localhost')

    async def run_source(self, handler):
        with patch('backend.app.services.public_scraper.validate_url', new=AsyncMock()), \
             patch('backend.app.services.public_scraper.asyncio.sleep', new=AsyncMock()):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await PublicScraper(client).scrape('https://example.com/page')

    async def test_success(self):
        result = await self.run_source(lambda r: httpx.Response(404) if r.url.path == '/robots.txt'
                                       else httpx.Response(200, json={'real': 1}))
        self.assertEqual(result['data'], {'real': 1})

    async def test_robots_denial_prevents_page_request(self):
        paths = []
        def handler(request):
            paths.append(request.url.path)
            return httpx.Response(200, text='User-agent: *\nDisallow: /')
        with self.assertRaisesRegex(ScrapeError, 'disallowed'):
            await self.run_source(handler)
        self.assertEqual(paths, ['/robots.txt'])

    async def test_denial_and_rate_limit_not_retried(self):
        for status in (401, 403, 429):
            paths = []
            def handler(request):
                paths.append(request.url.path)
                return httpx.Response(404 if request.url.path == '/robots.txt' else status)
            with self.assertRaisesRegex(ScrapeError, str(status)):
                await self.run_source(handler)
            self.assertEqual(paths, ['/robots.txt', '/page'])

    async def test_size_limit(self):
        with self.assertRaisesRegex(ScrapeError, '2 MB'):
            await self.run_source(lambda r: httpx.Response(404) if r.url.path == '/robots.txt'
                                  else httpx.Response(200, content=b'x' * 2_000_001))

    async def test_redirect_limit(self):
        with self.assertRaisesRegex(ScrapeError, 'redirects'):
            await self.run_source(lambda r: httpx.Response(404) if r.url.path == '/robots.txt'
                                  else httpx.Response(302, headers={'Location': '/loop'}))


if __name__ == '__main__':
    unittest.main()
