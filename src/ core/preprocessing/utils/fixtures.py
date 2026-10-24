import unicodedata
import pandas as pd
from fuzzywuzzy import process


def match_fixture_teams(league_df: pd.DataFrame, fixture_df: pd.DataFrame) -> pd.DataFrame:
    """ Matches the parsed team names from FootyStats with the corresponding Football-Data team names.
        It uses levenshtein distance.

        :param league_df: (pd.DataFrame) dataframe containing original league data.
        :param fixture_df: (pd.DataFrame) dataframe containing scraped league data from football-data.
    """

    def remove_non_ascii(x: str) -> str:
        """ Removes non-ascii characters. """

        return unicodedata.normalize('NFKD', x).encode('ascii', errors='ignore').decode('ascii')

    # Compute unique league teams/
    unique_teams = set(pd.concat([league_df['Home'], league_df['Away']]).dropna().unique())

    def match_team(team_name: str) -> str:
        return process.extractOne(team_name, unique_teams)[0]

    def normalize_and_match(team_name):
        if pd.isna(team_name):
            return team_name

        normalized_name = remove_non_ascii(str(team_name)).strip()
        if not normalized_name:
            return ''
        return match_team(normalized_name)

    # Normalize populated names and match them with the corresponding league
    # teams. Empty entry cells remain empty for fixture-row validation.
    fixture_df['Home'] = fixture_df['Home'].apply(normalize_and_match)
    fixture_df['Away'] = fixture_df['Away'].apply(normalize_and_match)
    return fixture_df
