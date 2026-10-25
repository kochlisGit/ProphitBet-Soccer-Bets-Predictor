import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Dict, List, Optional, Tuple
from lxml import html
from tqdm import tqdm
from urllib.error import URLError
from urllib.parse import urljoin
from src.core.network.downloaders.downloader import LeagueDownloader
from src.core.leagues.league import LeagueConfig


class BetExplorerLeagueDownloader(LeagueDownloader):
    """ Downloader class for leagues scraped from "betexplorer" webpage.

        **NOTE: This scraper might not work for some leagues, and a league's URL might change in the future. **
    """

    def __init__(self):
        super().__init__()

        self._seasonal_url_xpath = '//*[@value]'
        self._main_xpath = '//a[contains(@href, "stage=") and translate(normalize-space(string(.)), "MAIN", "main") = "main"]'
        self._matches_xpath = '(//table)[1]/tr'
        self._date_xpath = './td[contains(@class, "h-text-right")]'
        self._teams_xpath = './td[contains(@class, "h-text-left")]//a[contains(@class, "in-match")]/span'
        self._goals_xpath =  './td[contains(@class, "h-text-center")]/a'
        self._odds_xpath = './/*[@data-odd]/@data-odd'

        self._headers ={'User-Agent': 'Mozilla/5.0'}

    def _get_seasonal_urls(self, league_config: LeagueConfig) -> Dict[int, str]:
        current_year = date.today().year
        response = requests.get(league_config.league.url, timeout=15.0, headers=self._headers)

        if response.status_code != 200:
            raise URLError(
                f'\nFailed to download league "{league_config.league}": wrong URL: "{league_config.league.url}"'
            )

        tree = html.fromstring(response.content)
        elements = tree.xpath(self._seasonal_url_xpath)

        season_urls = {}
        for element in elements:
            value = element.get('value')

            if value and '/football/' in value:
                year_str = element.text_content().strip().split('/')[0]

                if year_str.isdigit():
                    year_int = min(current_year, int(year_str))

                    if year_int >= league_config.start_year:
                        season_urls[year_int] = f'{urljoin(response.url, value)}results'
        return season_urls

    def _parse_date(self, match, season: int) -> Optional[str]:
        """ Helper function that parses a match date. """

        date_element = match.xpath(self._date_xpath)

        if not date_element:
            return None

        date_str = date_element[0].text_content().rstrip()

        if date_str == 'Today':
            return date.today().strftime('%d/%m/%Y')

        day, month, year = date_str.split('.')

        if len(year) < 4:
            year = season

        date_str = f'{day}/{month}/{year}'
        return date_str

    def _parse_teams(self, match) -> Tuple[Optional[str], Optional[str]]:
        """ Helper function that parses a home, away teams. """

        team_elements = match.xpath(self._teams_xpath)

        if len(team_elements) >= 2:
            home_team = team_elements[0].text_content()
            away_team = team_elements[1].text_content()
        else:
            home_team = away_team = None

        return home_team, away_team

    def _parse_result(self, match) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """ Helper function that parses the result of a match."""

        score_elements = match.xpath(self._goals_xpath)

        if score_elements:
            scores = score_elements[0].text_content().split(' ', maxsplit=1)[0].split(':')

            if len(scores) == 2:
                hg = int(scores[0])
                ag = int(scores[1])

                if hg > ag:
                    result = 'H'
                elif hg == ag:
                    result = 'D'
                else:
                    result = 'A'
            else:
                hg = ag = result = None
        else:
            hg = ag = result = None

        return hg, ag, result

    def _parse_odds(self, match) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """ Helper function that parses the odds of a match. """

        odds = match.xpath(self._odds_xpath)

        if len(odds) == 3:
            odd_1 = float(odds[0])
            odd_x = float(odds[1])
            odd_2 = float(odds[2])
        else:
            odd_1 = odd_x = odd_2 = None

        return odd_1, odd_x, odd_2

    def _download_dataframe(self, league_config: LeagueConfig) -> Optional[pd.DataFrame]:
        seasonal_urls = self._get_seasonal_urls(league_config=league_config)

        def download_fn(year: int) -> List[List]:
            """ Helper function that downloads and returns main league data in a dataframe.

                Args:
                    year (int): Current year of league.
            """

            # Extracting html content from BetExplorer.
            url = seasonal_urls.get(year, None)

            if not url:
                return []

            response = requests.get(url=url, timeout=15.0, headers=self._headers)

            # Constructing HTML tree.
            tree = html.fromstring(response.content)

            # Identifying main tab option (if exists) and requesting the results from the main tab.
            main_links = tree.xpath(self._main_xpath)

            if main_links:
                main_href = main_links[0].get('href')
                main_url = urljoin(response.url, main_href)
                response = requests.get(url=main_url, timeout=15.0, headers=self._headers)
                tree = html.fromstring(response.content)

            # Parsing match rows.
            match_rows = tree.xpath(self._matches_xpath)

            if not match_rows:
                return []

            # Iterating rows.
            matches = []
            for match in match_rows:
                date_str = self._parse_date(match=match, season=year)
                if date_str is None:
                    continue

                home_team, away_team = self._parse_teams(match=match)
                if home_team is None:
                    continue

                hg, ag, result = self._parse_result(match=match)

                if result is None:
                    continue

                odd_1, odd_x, odd_2 = self._parse_odds(match=match)
                matches.append([date_str, year, home_team, away_team, hg, ag, result, odd_1, odd_x, odd_2])
            return matches

        # Computing max concurrent workers.
        years = range(league_config.start_year, date.today().year + 1)
        max_workers = min(len(years), 4)

        # Iterating seasons concurrently and downloading a dataframe per season.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            parsed_matches = list(
                tqdm(
                    executor.map(download_fn, years),
                    total=len(years),
                    desc='Downloading Year'
                )
            )
        data = sum(parsed_matches, [])

        if not data:
            raise AssertionError(
                f'\nFailed to download league: {league_config.league} from year {league_config.start_year}.'
            )

        df = pd.DataFrame(data=data, columns=self._expected_columns)
        return df
