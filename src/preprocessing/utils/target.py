import numpy as np
import pandas as pd
from enum import Enum
from sklearn.preprocessing import OneHotEncoder


class TargetType(Enum):
    """ The supported target types. """

    RESULT = 'result'
    OVER_UNDER = 'over-under'


def construct_targets(df: pd.DataFrame, target_type: TargetType) -> np.ndarray:
    """ Constructs the dataset targets based on the selected classification task """

    if target_type == TargetType.RESULT:
        y = df['Result'].replace({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32)
    elif target_type == TargetType.OVER_UNDER:
        y = ((df['HG'] + df['AG']).ge(2.5)).replace({False: 0, True: 1}).to_numpy(dtype=np.int32)
    else:
        raise TypeError(f'Undefiend target type: "{target_type.name}"')

    return y


def one_hot_encode(y: np.ndarray, target_type: TargetType) -> np.ndarray:
    """ One-Hot encodes the provided targets. To ensure consistency,
        the target categories are fixed and depend on the target type.
    """

    if target_type == TargetType.RESULT:
        y_encoded = OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False).fit_transform(y.reshape(-1, 1))
    elif target_type == TargetType.OVER_UNDER:
        raise TypeError('OVER_UNDER targets do not support one-hot encoding, as it is binary classification task.')
    else:
        raise TypeError(f'Not supported target type: "{type(target_type)}"')

    return y_encoded
