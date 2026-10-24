import math
import numpy as np
import pandas as pd
from typing import Generator, Optional, Tuple, Union
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def train_test_split(df: pd.DataFrame, test_size: Union[int, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Selects the first N matches as test set and the remaining matches as train set.

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe.
        :param test_size: (Union[int, float]) the size of the test set. If float is provided,
                          it will be converted to the corresponding number of test samples.
    """

    # Validate that the matches are provided in descending order, otherwise the calculations will be wrong.
    if not df['Date'].is_monotonic_decreasing:
        raise ValueError('Expected dates to be sorted in a descending order.')
    if df.isna().any().any():
        raise ValueError('Cannot split inputs to train-test with missing values. Drop rows with missing values first.')

    if isinstance(test_size, float):
        if not (0 < test_size <= 1.0):
            raise ValueError(f'test_ratio must be between 0 and 1, got "{test_size}"')

        test_size = int(math.floor(test_size*df.shape[0]))
    if test_size >= df.shape[0]:
        raise ValueError(f'test_size must be less than dataset size: {df.shape[0]}, got: {test_size}')

    df_test = df.iloc[:test_size].reset_index(drop=True)
    df_train = df.iloc[test_size:].reset_index(drop=True)
    return df_train, df_test


def cross_validation(
        df: pd.DataFrame,
        targets: Union[np.ndarray, pd.Series],
        k_folds: int = 5,
        show_progress: bool = False,
        seed: Optional[int] = 0
) -> Generator:
    """ Splits dataset using the Stratify k-fold cross validation method.
        Returns a tuple of: (fold, train dataframe, test dataframe).

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe.
        :param targets: (np.ndarray | pd.Series) the targets used for stratify sampling.
        :param k_folds: (int) number of k-folds (splits) to use.
        :param show_progress: (bool) whether to show the progress bar or not.
        :param seed: (int | None) random shuffling seed.
    """

    # Validate Dataframe.
    if df.isna().any().any():
        raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')

    cv_generator = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed).split(df, targets)
    iterator = enumerate(cv_generator)

    if show_progress:
        iterator = tqdm(iterable=iterator, desc='Evaluating K-Fold', total=k_folds)

    for i, (train_ids, eval_ids) in iterator:
        train_df = df.iloc[train_ids]
        eval_df = df.iloc[eval_ids]
        yield i, train_df, eval_df


def sliding_cross_validation(
        df: pd.DataFrame,
        test_ratio: float,
        k_folds: int = 5,
        show_progress: bool = False
) -> Generator:
    """ Splits dataset using the sliding cross validation method.
        Returns a tuple of: (fold, train dataframe, test dataframe).

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe.
        :param test_ratio: number of test proportion per fold. The proportion increases per fold.
        :param k_folds: (int) number of k-folds (splits) to use.
        :param show_progress: (bool) whether to show the progress bar or not.
    """

    # Validate Dataframe.
    if df.isna().any().any():
        raise ValueError('Cannot apply cross validation with nan rows. Drop nans first.')
    if not df['Date'].is_monotonic_decreasing:
        raise ValueError('Expected dates to be sorted in a descending order.')

    samples_per_fold = int(math.floor(df.shape[0]/k_folds))
    iterator = range(k_folds)

    if show_progress:
        iterator = tqdm(iterable=iterator, desc='Evaluating Sliding K-Fold', total=k_folds)

    for i in iterator:
        fold_df = df.iloc[-(i+1)*samples_per_fold:]
        train_df, eval_df = train_test_split(df=fold_df, test_size=test_ratio)
        yield i, train_df, eval_df
