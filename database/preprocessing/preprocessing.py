import pandas as pd
import numpy as np
import tensorflow.keras.utils as utils


def preprocess_data(results_and_stats: pd.DataFrame, one_hot: bool) -> (np.ndarray, np.ndarray):
    results_and_stats = results_and_stats.dropna()
    results_and_stats = results_and_stats.replace({'Result': {'H': 0, 'D': 1, 'A': 2}})
    targets = results_and_stats['Result'].to_numpy()
    if one_hot:
        targets = utils.to_categorical(targets)

    inputs = results_and_stats.drop(['Date', 'Home Team', 'Away Team', 'HG', 'AG', 'Result'], axis=1).to_numpy()
    return inputs, targets


def predictions_to_result(predictions: np.ndarray) -> list:
    result_str = {0: 'H', 1: 'D', 2: 'A'}
    return [result_str[pred] for pred in predictions]


def get_all_league_teams(results_and_stats: pd.DataFrame) -> set:
    return set(results_and_stats['Home Team'].unique()).union(set(results_and_stats['Away Team'].unique()))


def construct_input(
        results_and_stats: pd.DataFrame,
        home_team,
        away_team,
        odd_1,
        odd_x,
        odd_2
) -> np.ndarray:
    home_team_row = results_and_stats[results_and_stats['Home Team'] == home_team].head(1)
    away_team_row = results_and_stats[results_and_stats['Away Team'] == away_team].head(1)
    return np.hstack((
        np.float64([odd_1, odd_x, odd_2]),
        home_team_row[['HW', 'HL', 'HGF', 'HGD-W', 'HGD-L', 'HW%', 'HD%']].to_numpy().flatten(),
        away_team_row[['AW', 'AL', 'AGF', 'AGD-W', 'AGD-L', 'AW%', 'AD%']].to_numpy().flatten()
    ))
