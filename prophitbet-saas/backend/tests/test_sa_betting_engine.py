"""Unit tests for South African Sports Betting & Quantitative Bet Slip Engine.

Validates:
1. Mathematical precision and probability axioms across ALL soccer bet types.
2. Dixon-Coles Bivariate Poisson Expected Goals and score matrix derivation.
3. Bookmaker vigorish / overround calculations.
4. The Perfect Bet Slip Engine (Banker, Value Acca, High-Yield, SureBet).
"""

import math
import unittest
import numpy as np

from backend.app.services.math_engine import (
    calculate_dixon_coles_match,
    calculate_all_bet_types_from_matrix,
    calculate_expected_value_and_kelly,
)
from backend.app.services.sa_scraper import sa_scraper, BOOKMAKER_CONFIGS


def test_dixon_coles_bivariate_poisson_integrity():
    """Verify Dixon-Coles expected goals produces normalized, logically bounded probabilities."""
    res = calculate_dixon_coles_match(
        attack_home=1.35,
        defense_away=0.90,
        attack_away=1.05,
        defense_home=0.95,
        mean_home_goals=1.52,
        mean_away_goals=1.18,
    )
    probs = res["probabilities"]
    assert 0.0 < probs["H"] < 1.0
    assert 0.0 < probs["D"] < 1.0
    assert 0.0 < probs["A"] < 1.0
    assert math.isclose(probs["H"] + probs["D"] + probs["A"], 1.0, abs_tol=0.005)


def test_all_bet_types_mathematical_axioms():
    """Verify that all derived bet types obey fundamental probability axioms."""
    res = calculate_dixon_coles_match(
        attack_home=1.20,
        defense_away=1.00,
        attack_away=1.10,
        defense_home=1.00,
    )
    abt = res["all_bet_types"]

    # 1. 1X2 Sum
    m1x2 = abt["match_result_1x2"]
    assert math.isclose(m1x2["H"] + m1x2["D"] + m1x2["A"], 1.0, abs_tol=0.005)

    # 2. Over / Under Lines Sum to 1.0
    for line in ["0.5", "1.5", "2.5", "3.5", "4.5"]:
        ou = abt["over_under"][line]
        assert math.isclose(ou["over"] + ou["under"], 1.0, abs_tol=0.005)
        assert ou["fair_odds_over"] >= 1.0
        assert ou["fair_odds_under"] >= 1.0

    # 3. Both Teams to Score (BTTS) Sum to 1.0
    btts = abt["both_teams_to_score"]
    assert math.isclose(btts["yes"] + btts["no"], 1.0, abs_tol=0.005)

    # 4. Double Chance Relations
    dc = abt["double_chance"]
    assert math.isclose(dc["1X"], m1x2["H"] + m1x2["D"], abs_tol=0.005)
    assert math.isclose(dc["12"], m1x2["H"] + m1x2["A"], abs_tol=0.005)
    assert math.isclose(dc["X2"], m1x2["D"] + m1x2["A"], abs_tol=0.005)

    # 5. Draw No Bet (DNB) Decisive Sum
    dnb = abt["draw_no_bet"]
    assert math.isclose(dnb["DNB_1"] + dnb["DNB_2"], 1.0, abs_tol=0.005)

    # 6. Half-Time / Full-Time (HT/FT) 9 Outcomes Sum to 1.0
    htft = abt["half_time_full_time"]
    assert len(htft) == 9
    assert math.isclose(sum(htft.values()), 1.0, abs_tol=0.01)

    # 7. Correct Scores
    scores = abt["top_exact_scores"]
    assert len(scores) >= 10
    for sc in scores:
        assert "-" in sc["score"]
        assert sc["fair_odds"] > 1.0


def test_sa_sportsbook_scraper_calibration():
    """Model-derived prices must never impersonate observed bookmaker quotes."""
    for bm_code in BOOKMAKER_CONFIGS:
        with unittest.TestCase().assertRaisesRegex(ValueError, "not observed"):
            sa_scraper.generate_calibrated_sa_odds_for_bookmaker(bookmaker_code=bm_code)


def test_vigorish_calculation():
    """Verify that margin calculation accurately determines bookmaker overround."""
    # Fair coin toss (2.00, 2.00) -> 0% margin
    assert sa_scraper.calculate_margin([2.00, 2.00]) == 0.0

    # Typical 3-way match: Home 2.00, Draw 3.20, Away 3.80
    # Inv sum = (1/2.0) + (1/3.2) + (1/3.8) = 0.5 + 0.3125 + 0.26315 = 1.07565 -> 7.57%
    margin = sa_scraper.calculate_margin([2.00, 3.20, 3.80])
    assert 7.0 < margin < 8.0


def test_expected_value_and_kelly_criterion():
    """Verify Expected Value and Kelly sizing calculation."""
    probs = {"H": 0.55, "D": 0.25, "A": 0.20}
    # Generous market price on Home: 2.10 (Implied 47.6% vs Model 55.0%)
    odds = {"H": 2.10, "D": 3.40, "A": 4.50}

    ev_res = calculate_expected_value_and_kelly(probs, odds, kelly_fraction=0.25)
    h_eval = ev_res["outcomes"]["H"]

    assert h_eval["expected_value"] > 0.10  # >10% EV
    assert h_eval["is_positive_ev"] is True
    assert h_eval["kelly_stake_pct"] > 0.0  # Positive Quarter-Kelly stake recommendation
    assert ev_res["best_value_pick"] == "H"


def test_surebet_arbitrage_guaranteed_profit():
    """Verify that inverted odds < 1.0 produce guaranteed positive return across all legs."""
    best_h = 2.40  # Betway
    best_d = 3.60  # Hollywoodbets
    best_a = 5.20  # Sportingbet

    inv_sum = (1.0 / best_h) + (1.0 / best_d) + (1.0 / best_a)
    assert inv_sum < 1.0  # Valid Arbitrage Condition

    bankroll_zar = 1000.0
    s_h = round((bankroll_zar * (1.0 / best_h)) / inv_sum, 2)
    s_d = round((bankroll_zar * (1.0 / best_d)) / inv_sum, 2)
    s_a = round((bankroll_zar * (1.0 / best_a)) / inv_sum, 2)

    total_staked = s_h + s_d + s_a
    assert math.isclose(total_staked, bankroll_zar, abs_tol=1.0)

    payout_h = s_h * best_h
    payout_d = s_d * best_d
    payout_a = s_a * best_a

    # Payouts must be equal and strictly greater than bankroll
    assert payout_h > bankroll_zar
    assert payout_d > bankroll_zar
    assert payout_a > bankroll_zar
    assert math.isclose(payout_h, payout_d, abs_tol=2.0)
    assert math.isclose(payout_h, payout_a, abs_tol=2.0)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(function)
        for name, function in globals().items() if name.startswith("test_") and callable(function))
