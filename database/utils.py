import pandas as pd
import tensorflow.keras.utils as utils
import numpy as np


def preprocess_data(results_and_stats, columns):
    data_df = pd.DataFrame(results_and_stats, columns=columns)
    data_df = data_df.replace({'Result': {'H': 0, 'D': 1, 'A': 2}})

    targets = utils.to_categorical(data_df['Result'].to_numpy())
    inputs = data_df.drop(['Date', 'Home Team', 'Away Team', 'HG', 'AG', 'Result'], axis=1).to_numpy()

    return inputs, targets


def predictions_to_result(predictions):
    results = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            results.append('H')
        elif predictions[i] == 1:
            results.append('D')
        elif predictions[i] == 2:
            results.append('A')
    return results


def get_all_league_teams(results_and_stats):
    all_teams = set()

    for match in results_and_stats:
        all_teams.add(match[1])
        all_teams.add(match[2])
    return all_teams


def get_last_team_stats(results_and_stats, columns, team_name, is_home):
    team_stats_column_indices = []
    team_stats = []

    for i, column_name in enumerate(columns):
        if is_home:
            if column_name[0] == 'H' and column_name != 'HG' and column_name != 'Home Team':
                team_stats_column_indices.append(i)
        else:
            if column_name[0] == 'A' and column_name != 'AG' and column_name != 'Away Team':
                team_stats_column_indices.append(i)

    for row in results_and_stats:
        if is_home:
            row_team_name = row[1]
        else:
            row_team_name = row[2]

        if row_team_name == team_name:
            for col_index in team_stats_column_indices:
                team_stats.append(row[col_index])
            break
    return team_stats


def construct_prediction_sample(
        results_and_stats,
        columns,
        home_team,
        away_team,
        odd_1,
        odd_x,
        odd_2
):
    home_stats = get_last_team_stats(results_and_stats, columns, home_team, True)
    away_stats = get_last_team_stats(results_and_stats, columns, away_team, False)
    sample = [odd_1, odd_x, odd_2] + home_stats + away_stats
    return np.float32([sample])
