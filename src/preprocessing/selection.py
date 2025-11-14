import math
import pandas as pd
from typing import Tuple, Union


def train_test_split(df: pd.DataFrame, test_size: Union[int, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Selects the first N matches as test set and the remaining matches as train set. """

    # Validate that the matches are provided in descending order, otherwise the calculations will be wrong.
    if not df['Date'].is_monotonic_decreasing:
        raise ValueError('Expected dates to be sorted in a descending order.')

    if isinstance(test_size, float):
        test_size = int(math.floor(test_size*df.shape[0]/100.0))

    df_test = df.iloc[:test_size].reset_index(drop=True)
    df_train = df.iloc[test_size:].reset_index(drop=True)
    return df_train, df_test
