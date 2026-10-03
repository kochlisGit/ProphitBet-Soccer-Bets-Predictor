"""Collect public bookmaker DOM evidence using Dots' engine, without an LLM.

Run with tools/dots/.venv/Scripts/python.exe. No login, price clicks or bets.
"""
import argparse
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from invisible_playwright import InvisiblePlaywright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'storage' / 'bookmaker-feed' / 'snapshot.json'
SOURCES = {'BETWAY': 'https://sports.betway.co.za/sport/soccer',
           'HOLLYWOODBETS': 'https://www.hollywoodbets.net/'}


def allowed(url):
    parsed = urlparse(url)
    response = requests.get(f'{parsed.scheme}://{parsed.netloc}/robots.txt', timeout=15)
    if response.status_code == 404:
        return True
    response.raise_for_status()
    rules = RobotFileParser()
    rules.parse(response.text.splitlines())
    return rules.can_fetch('ProphitBetDataCollector', url)


def write_log(entry):
    log_dir = OUTPUT.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / 'collector.jsonl', maxBytes=2_000_000, backupCount=3, encoding='utf-8')
    try:
        handler.emit(logging.LogRecord('scraper', logging.INFO, '', 0, json.dumps(entry), (), None))
    finally:
        handler.close()


def collect():
    started = time.monotonic()
    snapshot = {'schema_version': 1, 'collector': 'dots-dom-v1', 'sources': {}}
    with InvisiblePlaywright(headless=True, prep_recaptcha=False,
                             timezone='Africa/Johannesburg') as browser:
        for bookmaker, url in SOURCES.items():
            source_started = time.monotonic()
            source = {'url': url, 'status': 'unavailable', 'events': []}
            snapshot['sources'][bookmaker] = source
            page = browser.new_page()
            try:
                if not allowed(url):
                    source['status'] = 'robots_disallowed'
                    continue
                response = page.goto(url, wait_until='domcontentloaded', timeout=45000)
                if response.status != 200:
                    source['status'] = f'http_{response.status}'
                    continue
                if bookmaker == 'BETWAY':
                    page.locator('a[href*="eventId="] strong').first.wait_for(timeout=30000)
                    source.update(page.locator('body').evaluate('''body => ({
                        headers: Array.from(body.querySelectorAll('.event-market-select-0, .event-market-0')).map(e=>e.innerText),
                        clock: (body.innerText.match(/Current time:\\s*(\\d{2}:\\d{2}:\\d{2})/)||[])[1]||null,
                        browser_timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                        events: Array.from(body.querySelectorAll('details > div[id]')).filter(e=>e.querySelector('a[href*="eventId="] strong')).slice(0,500).map(e=>e.outerHTML)
                    })'''))
                    source['status'] = 'collected' if source['events'] else 'no_events'
                else:
                    page.get_by_text('Upcoming Soccer', exact=True).first.wait_for(timeout=30000)
                    page.get_by_text('Upcoming Soccer', exact=True).first.click()
                    # Wait for an explicit empty state or an event listing, not the homepage shell.
                    try:
                        page.get_by_text('No events available', exact=True).first.wait_for(timeout=15000)
                        source['status'] = 'no_events'
                    except Exception:
                        source['status'] = 'unsupported_markup'
                    source['message'] = 'No validated public soccer event adapter available for the returned page.'
                source['url'] = page.url
                source['observed_at'] = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                source['status'] = 'collection_failed'
                source['error_type'] = type(exc).__name__
            finally:
                source['duration_seconds'] = round(time.monotonic() - source_started, 3)
                page.close()
    snapshot['collected_at'] = datetime.now(timezone.utc).isoformat()
    snapshot['duration_seconds'] = round(time.monotonic() - started, 3)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT.parent / 'evidence'
    archive.mkdir(exist_ok=True)
    # Retain source evidence for quotes imported from earlier observations.
    (archive / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')).write_text(
        json.dumps(snapshot), encoding='utf-8')
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(snapshot), encoding='utf-8')
    os.replace(temporary, OUTPUT)
    for name, source in snapshot['sources'].items():
        write_log(dict(timestamp=snapshot['collected_at'], stage='collection', bookmaker=name,
            status=source['status'], event_blocks=len(source['events']),
            duration_seconds=source['duration_seconds'], error_type=source.get('error_type')))
    print(json.dumps({name: {'status': source['status'], 'events': len(source['events'])}
                      for name, source in snapshot['sources'].items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=int, default=0, help='Repeat interval in seconds; 0 runs once, minimum 300.')
    args = parser.parse_args()
    if args.interval and args.interval < 300:
        parser.error('The minimum repeat interval is 300 seconds.')
    while True:
        try:
            collect()
        except Exception as exc:
            write_log(dict(timestamp=datetime.now(timezone.utc).isoformat(), stage='collection',
                           status='error', error_type=type(exc).__name__))
            raise
        if not args.interval:
            break
        time.sleep(args.interval)
