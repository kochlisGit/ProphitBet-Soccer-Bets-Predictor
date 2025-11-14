import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from imblearn.base import BaseSampler
from sklearn.base import TransformerMixin
from src.preprocessing.utils.normalization import NormalizerType, normalize
from src.preprocessing.utils.sampling import SamplerType, sample
from src.preprocessing.utils.target import TargetType, construct_targets


class DatasetPreprocessor:
    """ Dataset loader class. It has multiple purposes, including:
        1) Imputing missing data.
        2) Constructing Inputs & Targets based on the selected task.
        3) Normalizing inputs.
        4) Sampling inputs.
     """

    def __init__(self, drop_week: bool = True):
        """ Whether to drop week column. It is required to train the DRL agents. """

        self._non_trainable_columns = ['Date', 'Season', 'Home', 'Away', 'HG', 'AG', 'Result', 'Result-U/O', 'HST', 'AST', 'HC', 'AC']

        if drop_week:
            self._non_trainable_columns.append('Week')

    @property
    def non_trainable_columns(self) -> List[str]:
        return self._non_trainable_columns

    def preprocess_dataset(
            self,
            df: pd.DataFrame,
            target_type: TargetType,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[Union[SamplerType, BaseSampler]] = None,
            seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[TransformerMixin]]:
        """
            Preprocesses the dataframe and returns ready-to-train dataset, consisting of (input, target) pairs.
            :param df: The provided dataframe with the league matches.
            :param target_type: The selected target type of the dataset.
            :param normalizer: The normalization method of the input data.
            :param sampler: The sampling method of the input data.
            :param one_hot_targets: Whether to one-hot encode the target data.
            :param seed: The random seed of the sampler.
            :return: A tuple of: inputs (np.ndarray), targets (np.ndarray), normalizer (TransformerMixin)
        """

        df = df.dropna()

        # Construct inputs.
        x = df.drop(columns=self._non_trainable_columns, errors='ignore').to_numpy(dtype=np.float32)

        # Construct targets.
        y = construct_targets(df=df, target_type=target_type)

        # Apply input normalization and sampling.
        if normalizer is not None:
            x, normalizer = normalize(x=x, normalizer=normalizer)
        if sampler is not None:
            x, y, sampler = sample(x=x, y=y, sampler=sampler, seed=seed)

        # Validate (input, target) pair sizes.
        if x.shape[0] != y.shape[0]:
            raise ValueError(f'Found inconsistent sizes between input and target pairs: {x.shape[0]} vs {y.shape[0]}')

        return x, y, normalizer
