import numpy as np
from enum import Enum
from typing import Tuple, Union
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin


class NormalizerType(Enum):
    """ The supported normalization methods. """

    MIN_MAX = 'min-max'
    MAX_ABS = 'max-abs'
    STANDARD = 'standard'


def get_normalizer(normalizer: NormalizerType) -> TransformerMixin:
    """ Normalizer method factory. """

    if normalizer == NormalizerType.MIN_MAX:
        return MinMaxScaler()
    elif normalizer == NormalizerType.MAX_ABS:
        return MaxAbsScaler()
    elif normalizer == NormalizerType.STANDARD:
        return StandardScaler()
    else:
        raise NotImplementedError(f'Not implemented normalization method: "{normalizer.name}"')


def normalize(x: np.ndarray, normalizer: Union[NormalizerType, TransformerMixin]) -> Tuple[np.ndarray, TransformerMixin]:
    """ Normalizes the input dataframe and returns the normalized dataframe along with the normalization instance. """

    if isinstance(normalizer, NormalizerType):
        normalizer = get_normalizer(normalizer=normalizer)
        x_normalized = normalizer.fit_transform(x)
    elif isinstance(normalizer, TransformerMixin):
        x_normalized = normalizer.transform(x)
    else:
        raise TypeError(f'Not supported normalizer type: "{type(normalizer)}"')

    return x_normalized, normalizer
