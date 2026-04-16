import json
import logging
import time
import pandas as pd
from typing import Optional
from lxml import html
from selenium.webdriver import Chrome, Firefox, Edge, ChromeOptions, FirefoxOptions, EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from src.network.netutils import check_internet_connection


class FootyStatsScraper:
    """ FootyStats scraper, which opens FootyStats webpage via a web browser and parses the fixture table. """

    def __init__(self):
        self._page_load_timeout = 5.0
        self._poll_frequency = 0.5

        with open('storage/network/browser.json', mode='r') as jsonfile:
            browser = json.load(jsonfile)['application']

        if browser == 'chrome':
            options = ChromeOptions()
            options.add_argument('--incognito')
            options.add_argument('--lang=en-US')
            self._web_driver = Chrome(options=options)
        elif browser == 'firefox':
            options = FirefoxOptions()
            options.add_argument('--incognito')
            options.set_preference('intl.accept_languages', 'en-US, en')
            self._web_driver = Firefox(options=options)
        elif browser == 'edge':
            options = EdgeOptions()
            options.add_argument('--incognito')
            options.add_argument('--lang=en-US')
            self._web_driver = Edge(options=options)
        else:
            raise NotImplementedError(
                f'Not Implemented browser: "{browser}". '
                f'Only Chrome, Firefox and Edge are currently supported.'
            )

    def load_page(self, fixture_url: str) -> bool:
        """ Loads the FootyStats webpage and waits until loading state is ready. """

        # Check internet connection first.
        if not check_internet_connection():
            return False

        # Load webpage using the web driver.
        self._web_driver.get(url=fixture_url)
        self._web_driver.refresh()

        try:
            WebDriverWait(self._web_driver, timeout=self._page_load_timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.full-matches-table'))
            )
        except Exception as _:
            logging.info('Timed out waiting for fixture table to load.')
            return False

        time.sleep(1.0)
        return True

    def parse_fixture_table(self, date_str: str) -> Optional[pd.DataFrame]:
        """ Parses the fixture table. The fixture table should be displayed on the web page! """

        # Reads the odd from the provided span.
        def get_odd(span) -> str:
            text = span.text

            # If text is inside child element, it attempts to read the text inside the child element instead.
            # If no odd text is found, it returns 1.0, which id the default odd value.
            if text is None:
                for child in span:
                    text = child.text

                    if text is not None:
                        break

            return text.replace('\n', '').replace('\t', '') if text is not None else '1.0'

        tree = html.fromstring(self._web_driver.page_source)
        table_elements = tree.xpath('//div[contains(@class, "full-matches-table mt1e")]')

        if len(table_elements) == 0:
            raise RuntimeError('Could not find "full-matches-table mt1e" table class.')

        # Searching the requested table by date.
        formatted_date_str = f'{date_str} ~'
        requested_table = None
        for table in table_elements:
            date_element = table.find('h2')

            if date_element is None:
                continue

            # if date_element.text == formatted_date_str:
            if date_element.text_content().strip() == formatted_date_str:
                requested_table = table
                break

        if requested_table is None:
            logging.info(f'Could not find the selected date: "{date_str}" in a table header.')
            return None

        # Parsing fixture table.
        home_teams = []
        away_teams = []
        odds_1 = []
        odds_x = []
        odds_2 = []
        for ul in requested_table.findall('.//ul')[1:]:
            # Parsing teams.
            home_teams.append(ul.findall('.//a')[0].find('.//span').text)
            away_teams.append(ul.findall('.//a')[2].find('.//span').text)

            # Parsing odds.
            odd_spans = ul.findall('li')[-1].xpath('.//span[contains(@class, "hover-modal-parent")]')
            odd_1 = get_odd(span=odd_spans[0])
            odds_1.append(odd_1)
            odd_x = get_odd(span=odd_spans[1])
            odds_x.append(odd_x)
            odd_2 = get_odd(span=odd_spans[2])
            odds_2.append(odd_2)

        # Add year to dates.
        df = pd.DataFrame({
            'Home': home_teams,
            'Away': away_teams,
            '1': odds_1,
            'X': odds_x,
            '2': odds_2
        })
        return df

    def quit(self):
        self._web_driver.quit()
