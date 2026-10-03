import numpy as np
import pandas as pd
from enum import Enum
from sklearn.preprocessing import OneHotEncoder


class TargetType(Enum):
    """ The supported target types. """

    RESULT = 'result'
    OVER_UNDER = 'over-under'
    GOALS_15 = 'goals-1.5'
    GOALS_35 = 'goals-3.5'
    GOALS_45 = 'goals-4.5'
    BTTS = 'btts'
    HOME_GOALS_05 = 'home-goals-0.5'
    HOME_GOALS_15 = 'home-goals-1.5'
    AWAY_GOALS_05 = 'away-goals-0.5'
    AWAY_GOALS_15 = 'away-goals-1.5'
    CORNERS_85 = 'corners-8.5'
    CORNERS_95 = 'corners-9.5'
    CORNERS_105 = 'corners-10.5'
    CORNERS_115 = 'corners-11.5'
    SHOTS_TARGET_75 = 'shots-target-7.5'
    SHOTS_TARGET_85 = 'shots-target-8.5'


def parse_target(value):
    return value if isinstance(value, TargetType) else TargetType('over-under' if value == 'over_under' else value)


def target_columns(target_type):
    target = parse_target(target_type)
    if target == TargetType.RESULT:
        return ('Result',)
    if target.value.startswith('corners-'):
        return ('HC', 'AC')
    if target.value.startswith('shots-target-'):
        return ('HST', 'AST')
    if target.value.startswith('home-goals-'):
        return ('HG',)
    if target.value.startswith('away-goals-'):
        return ('AG',)
    return ('HG', 'AG')


def target_labels(target_type):
    target = parse_target(target_type)
    if target == TargetType.RESULT:
        return ['H', 'D', 'A']
    return ['No', 'Yes'] if target == TargetType.BTTS else ['Under', 'Over']


def valid_target_rows(df, target_type):
    """Unknown outcomes must never become negative-class examples."""
    columns = target_columns(target_type)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f'{parse_target(target_type).value} requires recorded columns: {sorted(missing)}')
    if parse_target(target_type) == TargetType.RESULT:
        return df['Result'].isin(['H', 'D', 'A'])
    values = df[list(columns)].apply(pd.to_numeric, errors='coerce')
    return (np.isfinite(values) & values.ge(0) & values.mod(1).eq(0)).all(axis=1)


def construct_targets(df: pd.DataFrame, target_type: TargetType) -> np.ndarray:
    """ Constructs the dataset targets based on the selected classification task """

    target_type = parse_target(target_type)
    if not valid_target_rows(df, target_type).all():
        raise ValueError(f'{target_type.value} contains missing or invalid observed outcomes')
    if target_type == TargetType.RESULT:
        y = df['Result'].replace({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32)
    else:
        values = df[list(target_columns(target_type))].apply(pd.to_numeric)
        if target_type == TargetType.BTTS:
            positive = values.gt(0).all(axis=1)
        else:
            line = 2.5 if target_type == TargetType.OVER_UNDER else float(target_type.value.rsplit('-', 1)[1])
            positive = values.sum(axis=1).gt(line)
        y = positive.to_numpy(dtype=np.int32)

    return y


def one_hot_encode(y: np.ndarray, target_type: TargetType) -> np.ndarray:
    """ One-Hot encodes the provided targets. To ensure consistency,
        the target categories are fixed and depend on the target type.
    """

    if target_type == TargetType.RESULT:
        y_encoded = OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False).fit_transform(y.reshape(-1, 1))
    elif isinstance(target_type, TargetType):
        raise TypeError('Binary targets do not support one-hot encoding.')
    else:
        raise TypeError(f'Not supported target type: "{type(target_type)}"')

    return y_encoded
