import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict


class TargetType(Enum):
    """ Supported target types that describe the classification tasks.

        Supported Types:
                1. RESULT: predict whether result will be H (Home Win), D (Draw) or A (Away Win).
                2. Result-U/O: predict under 2.5 or over 2.5 goals.
    """

    RESULT = 'Result'
    RESULT_UNDER_OVER = 'Result-U/O'

    def get_mapper(self) -> Dict[str, int]:
        """ Factory method that creates a new normalizer instance. """

        if self == TargetType.RESULT:
            return {'H': 0, 'D': 1, 'A': 2}
        elif self == TargetType.RESULT_UNDER_OVER:
            return {'U': 0, 'O': 1}
        else:
            raise NotImplementedError(f'Not supported target type: "{self.name}"')


def construct_targets(df: pd.DataFrame, target_type: TargetType, one_hot: bool = False) -> np.ndarray:
    """ Constructs dataset targets for the selected classification task.
         Returns a tuple of inputs (np.ndarray), normalizer (TransformerMixin)

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe.
        :param target_type: (TargetType) target type that describes the classification tasks.
        :param one_hot: (bool) whether to one-hot encode the targets. It is applied if number of targets > 2.
    """

    mapper = target_type.get_mapper()

    # Converting targets to numerical values.
    y = df[target_type.value].map(mapper).to_numpy(np.int32)

    # One-hot encoding targets.
    if one_hot:
        num_classes = len(mapper)

        if num_classes < 2:
            raise ValueError(f'One-hot encoding expects num_classes to be at least 2, got {num_classes}')

        y = np.eye(num_classes, dtype=np.float32)[y]

    return y
