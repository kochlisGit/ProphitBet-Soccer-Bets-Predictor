import numpy as np


def compute_profit_balance(odds: np.ndarray) -> float:
    normalized_sum = (1.0/odds).sum()
    normalized_factor = (1.0 - (1.0/normalized_sum))*100
    normalized_odds = (100.0 - normalized_factor)/odds
    normalized_profit = normalized_odds*(odds - 1.0)
    normalized_profit_balance = normalized_profit.sum()/100.0 + 1.0
    profit_balance = 1/normalized_profit_balance
    return round(profit_balance, 2)
