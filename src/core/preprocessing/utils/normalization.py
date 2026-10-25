import numpy as np
from enum import Enum
from typing import Tuple, Union
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin


class NormalizerType(Enum):
    """ The supported normalization methods.

        Supported Types:
                1. Min-Max Normalization: normalizes features in range [0.0, 1.0].
                2. Max-Abs Normalization: normalizes positive features in range [0.0, 1.0] and negative in [-1.0, 0.0].
                3. Standard Normalization: computes z-scores of each feature.
    """

    MIN_MAX = 'Min-Max'
    MAX_ABS = 'Max-Abs'
    STANDARD = 'Standard'

    @classmethod
    def from_str(cls, value: str) -> "NormalizerType":
        """ Converts string labels or names into NormalizerType.

            Args
            ----
            value: string value of the normalization method.
        """

        try:
            return cls(value)
        except ValueError:
            raise ValueError(f'Not supported normalization method: "{value}".')

    def create(self) -> TransformerMixin:
        """ Factory method that creates a new normalizer instance. """

        if self == NormalizerType.MIN_MAX:
            return MinMaxScaler()
        elif self == NormalizerType.MAX_ABS:
            return MaxAbsScaler()
        elif self == NormalizerType.STANDARD:
            return StandardScaler()
        else:
            raise NotImplementedError(f'Not supported normalization method: "{self.name}"')

def normalize_inputs(
        x: np.ndarray,
        normalizer: Union[str, NormalizerType, TransformerMixin]
) -> Tuple[np.ndarray, TransformerMixin]:
    """ Normalizes the input array and returns the normalized inputs and the fitted normalizer instance

        Args
        ----
        :param x: np.ndarray input array.
        :param normalizer: (str | NormalizerType | TransformerMixin) scikit-learn normalizer instance or
                           NormalizerType instance.
    """

    if isinstance(normalizer, str):
        normalizer = NormalizerType.from_str(normalizer)
    if isinstance(normalizer, NormalizerType):
        normalizer = normalizer.create()
        normalizer.fit(x)

    x = normalizer.transform(x)
    return x, normalizer
