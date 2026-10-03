"""Bounded public-page scraper and CLI. Output is unverified source data, not odds.

Run from prophitbet-saas:
python -m backend.app.services.public_scraper https://example.com --field title=h1
Use --rows '.event' --field 'home=.home' --field 'link=a::href' for records.
JavaScript-only pages need a site-specific adapter; access denials are not bypassed.
This local CLI is not an arbitrary-URL HTTP endpoint.
"""
import argparse
import asyncio
import hashlib
import ipaddress
import json
import socket
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

USER_AGENT = 'ProphitBetPublicScraper/1.0'
MAX_BYTES = 2_000_000


class ScrapeError(ValueError):
    """A source could not be safely retrieved or parsed."""


async def validate_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
        raise ScrapeError('Use a public HTTP(S) URL without embedded credentials.')
    if parts.port not in (None, 80, 443):
        raise ScrapeError('Only standard web ports are supported.')
    addresses = await asyncio.to_thread(socket.getaddrinfo, parts.hostname, parts.port or (443 if parts.scheme == 'https' else 80))
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ScrapeError('Local, private and reserved network addresses are not supported.')


class PublicScraper:
    def __init__(self, client):
        self.client = client
        self.policies = {}
        self.last_request = 0.0
        self.delay = 1.0

    async def _request(self, url):
        await validate_url(url)
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0, self.last_request + self.delay - loop.time()))
        self.last_request = loop.time()
        async with self.client.stream('GET', url, follow_redirects=False,
                                      headers={'User-Agent': USER_AGENT}, timeout=20) as response:
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    raise ScrapeError('Response exceeds the 2 MB limit.')
            return response.status_code, response.headers, bytes(body)

    async def _robots(self, url):
        parts = urlsplit(url)
        origin = f'{parts.scheme}://{parts.netloc}'
        if origin not in self.policies:
            status, _, body = await self._request(origin + '/robots.txt')
            policy = RobotFileParser()
            if status == 404:
                policy.parse([])
            elif status == 200:
                policy.parse(body.decode('utf-8', errors='replace').splitlines())
            else:
                raise ScrapeError(f'Cannot establish robots policy (HTTP {status}); stopping.')
            self.policies[origin] = policy
        policy = self.policies[origin]
        if not policy.can_fetch(USER_AGENT, url):
            raise ScrapeError('This path is disallowed by robots.txt.')
        self.delay = max(1.0, float(policy.crawl_delay(USER_AGENT) or 0))
        rate = policy.request_rate(USER_AGENT)
        if rate and rate.requests > 0:
            self.delay = max(self.delay, rate.seconds / rate.requests)
        if self.delay > 60:
            raise ScrapeError('Source requires a longer crawl interval; use a scheduled site-specific adapter.')

    async def scrape(self, url, *, rows=None, fields=None):
        try:
            for _ in range(6):
                await validate_url(url)
                await self._robots(url)
                status, headers, body = await self._request(url)
                if status in (301, 302, 303, 307, 308):
                    if not headers.get('location'):
                        raise ScrapeError('Redirect has no destination.')
                    url = urljoin(url, headers['location'])
                    continue
                if status in (401, 403, 429):
                    raise ScrapeError(f'HTTP {status}: access denied or rate limited; no bypass attempted.')
                if status != 200:
                    raise ScrapeError(f'Source returned HTTP {status}.')
                break
            else:
                raise ScrapeError('Too many redirects.')
            return parse_page(url, headers.get('content-type', ''), body, rows=rows, fields=fields)
        except (httpx.HTTPError, OSError) as exc:
            raise ScrapeError(f'Request failed ({type(exc).__name__}).') from exc


def parse_page(url, content_type, body, *, rows=None, fields=None):
    result = dict(source_url=url, fetched_at=datetime.now(timezone.utc).isoformat(),
                  sha256=hashlib.sha256(body).hexdigest(), verified=False)
    media = content_type.split(';')[0].strip().lower()
    if media == 'application/json' or media.endswith('+json'):
        if rows or fields:
            raise ScrapeError('CSS extraction applies only to HTML, not JSON.')
        try:
            result.update(format='json', data=json.loads(body))
        except (ValueError, UnicodeError) as exc:
            raise ScrapeError('Source contains invalid JSON.') from exc
    elif media in ('text/html', 'application/xhtml+xml'):
        soup = BeautifulSoup(body, 'html.parser')
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        if rows or fields:
            records = []
            try:
                for node in soup.select(rows) if rows else [soup]:
                    record = {}
                    for name, spec in (fields or {'text': ':scope'}).items():
                        selector, separator, attribute = spec.partition('::')
                        found = node if selector == ':scope' else node.select_one(selector)
                        record[name] = (found.get(attribute) if separator else found.get_text(' ', strip=True)) if found is not None else None
                    records.append(record)
            except Exception as exc:
                raise ScrapeError('Invalid CSS selector or extraction rule.') from exc
            result.update(format='html', records=records, count=len(records))
        else:
            result.update(format='html', title=soup.title.get_text(' ', strip=True) if soup.title else None,
                          text=soup.get_text(' ', strip=True),
                          links=[dict(text=a.get_text(' ', strip=True), url=urljoin(url, a['href']))
                                 for a in soup.select('a[href]') if urlsplit(urljoin(url, a['href'])).scheme in ('http', 'https')])
    else:
        raise ScrapeError('Unsupported content type; only HTML and JSON are supported.')
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('--rows', help='CSS selector for repeating records')
    parser.add_argument('--field', action='append', default=[], help='NAME=CSS or NAME=CSS::attribute')
    args = parser.parse_args()
    fields = {}
    for item in args.field:
        name, sep, selector = item.partition('=')
        if not sep or not name or not selector or name in fields:
            parser.error('Each field must be a unique NAME=CSS extraction rule.')
        fields[name] = selector
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            result = await PublicScraper(client).scrape(args.url, rows=args.rows, fields=fields)
        except (ScrapeError, ValueError) as exc:
            parser.exit(1, f'Scraping stopped: {exc}\n')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
