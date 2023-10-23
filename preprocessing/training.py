import pandas as pd
import numpy as np
import tensorflow.keras.utils as utils


def preprocess_training_dataframe(matches_df: pd.DataFrame, one_hot: bool) -> (np.ndarray, np.ndarray):
    inputs = matches_df.dropna().drop(columns=['Season', 'Date', 'Result', 'Home Team', 'Away Team'])
    inputs = inputs.to_numpy(dtype=np.float64)
    targets = matches_df['Result'].replace({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int64)

    if one_hot:
        targets = utils.to_categorical(targets)

    return inputs, targets


def construct_input_from_team_names(
        matches_df: pd.DataFrame,
        home_team: str,
        away_team: str,
        odd_1: float,
        odd_x: float,
        odd_2: float
) -> np.ndarray:
    home_team_row = matches_df[matches_df['Home Team'] == home_team].head(1).drop(
        columns=['Season', 'Date', 'Result', 'Home Team', 'Away Team']
    )
    away_team_row = matches_df[matches_df['Away Team'] == away_team].head(1).drop(
        columns=['Season', 'Date', 'Result', 'Home Team', 'Away Team']
    )
    return np.hstack((
        np.float64([odd_1, odd_x, odd_2]),
        home_team_row[[col for col in home_team_row.columns if col[0] == 'H']].to_numpy(dtype=np.float64).flatten(),
        away_team_row[[col for col in home_team_row.columns if col[0] == 'A']].to_numpy(dtype=np.float64).flatten()
    )).reshape((1, -1))


def construct_inputs_from_fixtures(
        matches_df: pd.DataFrame,
        fixtures_df: pd.DataFrame
) -> np.ndarray:
    return np.vstack([
        construct_input_from_team_names(
            matches_df=matches_df,
            home_team=match['Home Team'],
            away_team=match['Away Team'],
            odd_1=match['1'],
            odd_x=match['X'],
            odd_2=match['2']
        )
        for _, match in fixtures_df.iterrows()
    ])


def split_train_targets(
        inputs: np.ndarray,
        targets: np.ndarray,
        num_eval_samples: int
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    x_train = inputs[num_eval_samples:]
    y_train = targets[num_eval_samples:]
    x_test = inputs[: num_eval_samples]
    y_test = targets[: num_eval_samples]
    return x_train, y_train, x_test, y_test
