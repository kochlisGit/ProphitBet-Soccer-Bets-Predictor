import math
import pandas as pd
from typing import Tuple, Union


def train_test_split(df: pd.DataFrame, test_size: Union[int, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Selects the first N matches as test set and the remaining matches as train set. """

    # Validate that the matches are provided in descending order, otherwise the calculations will be wrong.
    if not df['Date'].is_monotonic_decreasing:
        raise ValueError('Expected dates to be sorted in a descending order.')

    if isinstance(test_size, float):
        if test_size <= 0:
            raise ValueError('test_size must be greater than zero.')
        # Accept both sklearn-style ratios (0.2) and the desktop UI's
        # percentage values (20.0). The old implementation treated 0.2 as
        # 0.2%, producing an empty evaluation set for SaaS model training.
        fraction = test_size if test_size <= 1 else test_size / 100.0
        test_size = max(1, int(math.floor(fraction * df.shape[0])))

    if test_size <= 0 or test_size >= df.shape[0]:
        raise ValueError('test_size must leave at least one training and one evaluation row.')

    # Keep all matches on the boundary date together: same-day results must
    # never leak from training into evaluation when kickoff times are unknown.
    dates = pd.to_datetime(df['Date'], errors='raise', utc=True).dt.normalize()
    if dates.isna().any():
        raise ValueError('Missing match dates cannot be split chronologically.')
    boundary = dates.iloc[test_size - 1]
    test_size = int((dates >= boundary).sum())
    if test_size >= len(df):
        raise ValueError('Need distinct dates for training and evaluation.')
    df_test = df.iloc[:test_size].reset_index(drop=True)
    df_train = df.iloc[test_size:].reset_index(drop=True)
    return df_train, df_test
