class FixtureParser:
    def __init__(self):
        self._table_id = 'full-matches-table mt1e'
        self._team_id = 'data-comp-id'
        self._odds_id = 'hover-modal-parent">'

    @property
    def table_id(self) -> str:
        return self._table_id

    @property
    def team_id(self) -> str:
        return self._team_id

    @property
    def odds_id(self) -> str:
        return self._odds_id

    def _parse_matches(self, teams_lines: list, n_teams: int) -> list:
        matches = []

        for i in range(0, n_teams, 2):
            home_team = teams_lines[i].split('>', 1)[1].split('<', 1)[0]
            away_team = teams_lines[i+1].split('>', 1)[1].split('<', 1)[0]
            matches.append((home_team, away_team))
        return matches

    def _parse_odds(self, odd_element_lines: list, n_matches: int) -> list:
        odds = []
        for i in range(0, n_matches*3, 3):
            odd_1 = float(odd_element_lines[i].split('\n<span', 1)[0])
            odd_x = float(odd_element_lines[i+1].split('\n<span', 1)[0])
            odd_2 = float(odd_element_lines[i+2].split('\n<span', 1)[0])
            odds.append((odd_1, odd_x, odd_2))
        return odds

    def parse_fixture(self, fixture_filepath: str) -> (list, list):
        upcoming_matches = []
        odds = []

        with open(fixture_filepath, 'r', encoding='utf-8') as fixture_html:
            lines = fixture_html.read()

        match_table_lines = lines.split(self.table_id)
        match_table_str = None if len(match_table_lines) < 2 else match_table_lines[1]

        if match_table_str is not None:
            teams_lines = match_table_str.split(self.team_id)[1:]
            n_teams = len(teams_lines)

            if len(teams_lines) > 0:
                upcoming_matches = self._parse_matches(teams_lines=teams_lines, n_teams=n_teams)
                n_upcoming_matches = len(upcoming_matches)
                odd_element_lines = match_table_str.split(self.odds_id)[1:]

                if len(odd_element_lines) >= n_upcoming_matches * 3:
                    odds = self._parse_odds(odd_element_lines=odd_element_lines, n_matches=n_upcoming_matches)
        return upcoming_matches, odds
