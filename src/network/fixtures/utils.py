import pandas as pd
from fuzzywuzzy import process


def match_fixture_teams(parsed_teams_df: pd.DataFrame, league_df: pd.DataFrame) -> pd.DataFrame:
    """ Matches the parsed team names from FootyStats with the corresponding Football-Data team names. """

    # Compute unique league teams/
    unique_teams = set(pd.concat([league_df['Home'], league_df['Away']]).unique())

    def match_team(team_name: str) -> str:
        return process.extractOne(team_name, unique_teams)[0]

    # Match parsed teams with the corresponding league teams.
    parsed_teams_df['Home'] = parsed_teams_df['Home'].apply(match_team)
    parsed_teams_df['Away'] = parsed_teams_df['Away'].apply(match_team)
    return parsed_teams_df
