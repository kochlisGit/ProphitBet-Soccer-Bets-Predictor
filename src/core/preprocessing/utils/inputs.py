import pandas as pd
from typing import Optional


def construct_inputs_by_match(league_df: pd.DataFrame, match_df: pd.Series) -> Optional[pd.Series]:
    """ Constructs a model input using the home team, away team and the odds for a specified match.
        If no historical match is found between home and away teams, it returns None.

        Args
        ----
        :param league_df: (pd.DataFrame) league's dataframe.
        :param match_df: (pd.Series) match's series that includes home, away teams, and 1,X,2 odds.
    """

    home, away, odd_1, odd_x, odd_2 = match_df
    mask = (league_df['Home'] == home) & (league_df['Away'] == away)
    rows = league_df[mask]

    if rows.shape[0] == 0:
        return None

    row = rows.iloc[0]
    row[['1', 'X', '2']] = [odd_1, odd_x, odd_2]

    # Validating selected row.
    if row.isna().any():
        raise ValueError('Found missing values in rows. Remove missing values before calling this method.')

    return row


def construct_inputs_from_fixture(df: pd.DataFrame, fixture_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """ Constructs a model input using a fixture (multiple matches) that include home, away, 1, X, 2 odds.
        Missing values should be omitted before calling this method.

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe.
        :param fixture_df: (pd.Series) fixture's dataframe that includes home, away teams, and 1,X,2 odds.
    """

    # Ensuring nan rows are dropped.
    if df.isna().any().any():
        raise ValueError('Found missing values in dataframe. Remove missing values before calling this method.')

    # Constructing inputs for each match.
    inputs = fixture_df.apply(lambda match_df: construct_inputs_by_match(league_df=df, match_df=match_df), axis=1)

    # Validating inputs.
    valid_inputs = [x for x in inputs if x is not None]

    if not valid_inputs:
        x = None
    elif len(valid_inputs) == 1:
        x = valid_inputs[0].to_frame().T.reset_index(drop=True)
    else:
        x = pd.concat(valid_inputs, axis=0, ignore_index=True)

    return x
