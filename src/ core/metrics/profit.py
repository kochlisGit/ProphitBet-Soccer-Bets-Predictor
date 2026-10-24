import numpy as np


def mean_implied_probability(odds: np.ndarray) -> float:
    """ Computes mean implied probability for the given odds.  """

    odds = np.asarray(odds, dtype=np.float64)
    return (1.0/odds).mean()
