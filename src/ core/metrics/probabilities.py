import numpy as np


def convert_odds_to_probabilities(odds: np.ndarray) -> np.ndarray:
    """ Converts raw odds to probabilities.

        Args
        ----
        odds: (np.ndarray) raw odds values.
    """

    raw_probabilities = 1.0/odds
    probability_sum = raw_probabilities.sum(axis=1, keepdims=True)
    bookmaker_prob = raw_probabilities/probability_sum
    return bookmaker_prob


def ranked_probabilities_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """ Computes Ranked Probability Score.

        Args
        ----
        :param y_true: (np.ndarray) true labels.
        :param y_proba: (np.ndarray) model's predicted probabilities.
    """

    y_true_one_hot = np.eye(y_proba.shape[1])[y_true]
    cumulative_prob = np.cumsum(y_proba, axis=1)[:, :-1]
    cumulative_true = np.cumsum(y_true_one_hot, axis=1)[:, :-1]
    return np.mean((cumulative_prob - cumulative_true) ** 2)
