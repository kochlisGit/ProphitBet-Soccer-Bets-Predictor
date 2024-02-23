import time
import pandas as pd
import config
from selenium.webdriver import Chrome, Firefox, Edge, ChromeOptions, FirefoxOptions, EdgeOptions
from lxml import html


class FootyStatsScraper:
    def __init__(self, browser: str):
        if browser == 'chrome':
            options = ChromeOptions()
            options.add_argument('--incognito')
            options.add_argument("--lang=en-US")
            self._web_driver = Chrome(options=options)
        elif browser == 'firefox':
            options = FirefoxOptions()
            options.add_argument('--incognito')
            options.set_preference("intl.accept_languages", "en-US, en")
            self._web_driver = Firefox(options=options)
        elif browser == 'edge':
            options = EdgeOptions()
            options.add_argument('--incognito')
            options.add_argument("--lang=en-US")
            self._web_driver = Edge(options=options)
        else:
            raise NotImplementedError(f'Not Implemented Browser: "{browser}"')

    def parse_matches(self, fixtures_url: str, date_str: str) -> pd.DataFrame or str:
        self._web_driver.get(url=fixtures_url)
        time.sleep(1)

        tree = html.fromstring(self._web_driver.page_source)
        fixture_elements = tree.xpath(f"//*[contains(@class, '{config.fixtures_class}')]")

        if len(fixture_elements) == 0:
            self._web_driver.quit()
            return f'Couldn\'t find fixture elements with class="{config.fixtures_class}"'

        formatted_date_str = f'{date_str} ~'
        matches_element = None
        for element in fixture_elements:
            fixture_date_elements = element.find_class(config.fixture_date_class)

            if len(fixture_date_elements) == 0:
                self._web_driver.quit()
                return f'Couldn\'t find fixture date element with class="{config.fixture_date_class}"'
            else:
                fixture_date_str = fixture_date_elements[0].text

                if fixture_date_str == formatted_date_str:
                    matches_element = element
                    break

        if matches_element is None:
            self._web_driver.quit()
            return f'Couldn\'t find fixture elements with date: {date_str}'

        matches = []

        team_elements = matches_element.find_class(config.team_name_class)
        num_team_elements = len(team_elements)
        num_matches = num_team_elements//2

        if num_team_elements == 0:
            self._web_driver.quit()
            return f'Couldn\'t find names elements with class="{config.team_name_class}"'

        if num_team_elements % 2 != 0:
            self._web_driver.quit()
            return f'An odd number ({len(team_elements)}) of match elements is found present in the fixture: Some match elements are missing. Check the fixture.'

        odd_rows_elements = matches_element.find_class(config.odd_row_class)

        if odd_rows_elements == 0:
            self._web_driver.quit()
            return f'Couldn\'t find odd elements with class="{config.odd_row_class}"'
        if len(odd_rows_elements) != num_matches:
            self._web_driver.quit()
            return f'Expected {num_matches} odd rows elements, got {len(odd_rows_elements)}: Some odd elements are missing. Check the fixture.'

        for i in range(num_matches):
            j = i*2

            odd_elements = odd_rows_elements[i].xpath(f'span[contains(@class, "{config.odd_values_class}")]/text()')
            row = [
                team_elements[j].text,
                team_elements[j + 1].text,
                odd_elements[0].replace('\n', ''),
                odd_elements[2].replace('\n', ''),
                odd_elements[4].replace('\n', '')
            ]
            matches.append(row)
        self._web_driver.quit()
        return pd.DataFrame(data=matches, columns=['Home Team', 'Away Team', '1', 'X', '2'])
