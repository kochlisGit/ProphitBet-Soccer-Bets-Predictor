import re

import pandas as pd

from fixtures.similarities.matching import match_fixture_teams


class FootyStatsFixtureParser:
    def __init__(self):
        self._team_id = "data-comp-id"
        self._odds_id = " hover-modal-parent"
        self._num_teams = 20

    def get_available_match_tables(self, fixture_filepath):
        with open(fixture_filepath, "r", encoding="utf-8") as htmlfile:
            fixture_str = htmlfile.read()
        regex = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d+\s~'
        return re.findall(regex, fixture_str)

    def _get_match_table(self, fixture_str, fixture_date: str) -> str or None:
        tables = fixture_str.split(fixture_date)

        if len(tables) == 1:
            return None

        return tables[1]

    def _get_teams(self, table_lines: str) -> (list, list) or (None, None):
        home_teams = []
        away_teams = []

        team_lines = table_lines.split(self._team_id)
        num_lines = min(self._num_teams, len(team_lines))
        if num_lines == 1:
            return None, None

        for i in range(1, num_lines, 2):
            home_teams.append(team_lines[i].split(">", 1)[1].split("<", 1)[0])
            away_teams.append(team_lines[i + 1].split(">", 1)[1].split("<", 1)[0])
        return home_teams, away_teams

    def _get_odds(
        self, table_lines: str, num_matches: int
    ) -> (list, list, list) or (None, None, None):
        odds_1 = []
        odds_x = []
        odds_2 = []

        odd_lines = table_lines.split(self._odds_id)
        num_odd_lines = min((self._num_teams // 2) * 3, len(odd_lines))

        if num_odd_lines == 1:
            return None, None, None

        for i in range(1, (num_matches * 3) + 1, 3):
            odds_1.append(odd_lines[i].split(">")[1].split("<")[0].replace("\n", ""))
            odds_x.append(
                odd_lines[i + 1].split(">")[1].split("<")[0].replace("\n", "")
            )
            odds_2.append(
                odd_lines[i + 2].split(">")[1].split("<")[0].replace("\n", "")
            )
        return odds_1, odds_x, odds_2

    def parse_fixture(
        self,
        fixture_filepath: str,
        fixture_date: str,
        unique_league_teams: set,
    ) -> pd.DataFrame or str:

        with open(fixture_filepath, "r", encoding="utf-8") as htmlfile:
            fixture_str = htmlfile.read()

        table_lines = self._get_match_table(
            fixture_str=fixture_str, fixture_date=fixture_date
        )

        if table_lines is None:
            return (
                f'Parsing Error: Failed to parse fixture table, because date "{fixture_date}" was not found. '
                f"Perhaps the selected date is incorrect or the HTML of Footystats code has changed."
            )

        home_teams, away_teams = self._get_teams(table_lines=table_lines)

        if home_teams is None or away_teams is None:
            return (
                f'Parsing Error: Failed to parse fixture teams, because keyword "{self._team_id}" was not found. '
                f"Perhaps the HTML of Footystats code has changed. The parser has to be updated: Contact developer."
            )

        num_teams = len(home_teams)
        odds_1, odds_x, odds_2 = self._get_odds(
            table_lines=table_lines, num_matches=num_teams
        )

        if odds_1 is None:
            odds_1 = odds_x = odds_2 = ["" for _ in range(num_teams)]

        home_teams, away_teams = match_fixture_teams(
            fix_home_teams=home_teams,
            fix_away_teams=away_teams,
            unique_league_teams=unique_league_teams,
        )
        return pd.DataFrame(
            {
                "Home Team": home_teams,
                "Away Team": away_teams,
                "1": odds_1,
                "X": odds_x,
                "2": odds_2,
            }
        )
