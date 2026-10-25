import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from imblearn.base import BaseSampler
from sklearn.base import TransformerMixin
from src.core.preprocessing.utils import (
    NormalizerType,
    normalize_inputs,
    SamplerType,
    sample_inputs,
    TargetType,
    construct_targets
)


class DatasetPreprocessor:
    """ Multipurpose dataset processing class.

        Purposes:
            1. Dropping rows with missing values.
            2. Normalizes inputs, constructs and encodes targets.
     """

    def __init__(self):
        self._metadata_columns = [
            'Date', 'Season', 'Week', 'Home', 'Away', '1X', '12', 'X2', 'HG', 'AG', 'Result', 'Result-U/O'
        ]

    @property
    def metadata_columns(self) -> List[str]:
        return self._metadata_columns

    def construct_inputs(
            self,
            df: pd.DataFrame,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None
    ) -> Tuple[np.ndarray, Optional[TransformerMixin]]:
        """ Construct dataset inputs. The same pipeline is applied for every classification task.

            Args
            ----
            :param df: (pd.DataFrame) league's dataframe.
            :param normalizer: (Normalizer) normalization method of the input data. Defaults to None.
        """

        # Construct inputs.
        x = df.drop(columns=self._metadata_columns).to_numpy(dtype=np.float32)

        # Normalizing inputs.
        if normalizer is not None:
            x, normalizer = normalize_inputs(x=x, normalizer=normalizer)

        return x, normalizer

    def preprocess(
            self,
            df: pd.DataFrame,
            target_type: TargetType,
            one_hot_targets: bool = False,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[Union[SamplerType, BaseSampler]] = None,
            seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[TransformerMixin]]:
        """ Preprocesses the dataframe and returns ready-to-train dataset, consisting of (input, target) pairs.
            Returns a tuple of: inputs (np.ndarray), targets (np.ndarray), normalizer (TransformerMixin)

            Args
            ----
            :param df: (pd.DataFrame) league's dataframe.
            :param target_type: (TargetType) specified target type that describes the classification task.
            :param one_hot_targets: (bool) whether to one-hot encode the target data.
            :param normalizer: (Normalizer) normalization method of the input data. Defaults to None.
            :param sampler: (Sampler) sampling method of the input data. Defaults to None
            :param seed: random seed of the sampler. Defaults to None.
            :return:
        """

        if df.isna().any().any():
            raise ValueError('Found missing values. Drop rows with missing values to continue.')

        x, normalizer = self.construct_inputs(df=df, normalizer=normalizer)
        y = construct_targets(df=df, target_type=target_type, one_hot=one_hot_targets)

        # Sampling inputs.
        if sampler is not None:
            x, y = sample_inputs(x=x, y=y, sampler=sampler, seed=seed)

        # Validate (input, target) pair sizes.
        if x.shape[0] != y.shape[0]:
            raise ValueError(f'Found inconsistent sizes between input and target pairs: {x.shape[0]} vs {y.shape[0]}')

        return x, y, normalizer
