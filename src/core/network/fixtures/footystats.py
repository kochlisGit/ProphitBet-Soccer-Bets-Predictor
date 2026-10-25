import requests
import pandas as pd
from typing import List, Optional
from lxml import html


class FootyStatsScraper:
    """ Footystats scraper class. It scrapes league data from each feature. """

    def __init__(self):
        self._headers ={'User-Agent': 'Mozilla/5.0'}

        self._xpath_date_query = '//div[contains(@class, "full-matches-table mt1e")]//h2/text()'
        self._xpath_table_query = '//div[contains(@class, "full-matches-table mt1e")]'

    def scrape(self, fixture_url: str):
        """ Scrapes HTML webpage from footystats.

            fixture_url: (str) footystat's fixture url.
        """

        # Scraping HTML webpage.
        response = requests.get(url=fixture_url, headers=self._headers)
        response.raise_for_status()

        # Constructing tree from HTML source code.
        tree = html.fromstring(response.content)
        return tree

    def get_available_table_dates(self, tree) -> Optional[List[str]]:
        """ Returns a list of all table dates provided by footystats.

            Args
            ----
            :param tree: Constructed tree (root) element.
        """

        # Parsing dates.
        date_contents = tree.xpath(self._xpath_date_query)

        if not date_contents:
            return None

        dates = [date_str[: -2] for date_str in date_contents]
        return dates

    def parse_league_table(self, tree, table_index: int) -> Optional[pd.DataFrame]:
        """ Parses the fixture table of the specified date. Table's format is: Home | Away | 1 | X | 2

            Args
            ----
            :param tree: Constructed tree (root) element.
            :param table_index: (str) specified table_index, which is extracted using the date.
        """

        # Reads the odd from the provided span.
        def get_odd(span=None) -> float:
            """ Parses the odd text. """

            if span is None:
                return 1.0

            # Odds can be direct text, nested text, or an empty placeholder.
            # Missing/unparseable odds retain the scraper's documented 1.0
            # sentinel so callers can request a user correction.
            text = span.text.strip()
            if not text:
                return 1.0

            try:
                return float(text)
            except ValueError:
                return 1.0

        # Scraping the table of the specified date.
        table_elements = tree.xpath(self._xpath_table_query)

        if not table_elements:
            return None

        requested_table = table_elements[table_index]
        home_teams = []
        away_teams = []
        odds_1 = []
        odds_x = []
        odds_2 = []
        odds_1x = []
        odds_12 = []
        odds_x2 = []
        for ul in requested_table.findall('.//ul')[1:]:
            # Parsing teams.
            home_teams.append(ul.findall('.//a')[0].find('.//span').text)
            away_teams.append(ul.findall('.//a')[2].find('.//span').text)

            # Parsing odds.
            odd_spans = ul.findall('li')[-1].xpath('.//span[contains(@class, "hover-modal-parent")]')
            num_spans = len(odd_spans)

            odd_1 = get_odd(span=odd_spans[0] if num_spans > 0 else None)
            odds_1.append(odd_1)
            odd_x = get_odd(span=odd_spans[1] if num_spans > 1 else None)
            odds_x.append(odd_x)
            odd_2 = get_odd(span=odd_spans[2]if num_spans > 2 else None)
            odds_2.append(odd_2)

            # Calculating double chance odds.
            odd_1_prob = 1.0/odd_1
            odd_x_prob = 1.0/odd_x
            odd_2_prob = 1.0/odd_2
            odd_1x = max(round(1.0/(odd_1_prob + odd_x_prob), 2), 1.0)
            odds_1x.append(odd_1x)
            odd_12 = max(round(1.0/(odd_1_prob + odd_2_prob), 2), 1.0)
            odds_12.append(odd_12)
            odd_x2 = max(round(1.0/(odd_x_prob + odd_2_prob), 2), 1.0)
            odds_x2.append(odd_x2)

        # Add year to dates.
        df = pd.DataFrame({
            'Home': home_teams,
            'Away': away_teams,
            '1': odds_1,
            'X': odds_x,
            '2': odds_2,
            '1X': odds_1x,
            '12': odds_12,
            'X2': odds_x2
        })
        return df
