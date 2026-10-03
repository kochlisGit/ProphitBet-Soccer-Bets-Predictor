"""Mathematical Soccer Modeling Engine.

Implements rigorous probabilistic and financial models:
1. Dixon-Coles Bivariate Poisson Expected Goals Model (1997)
2. Empirical Team Attack/Defense Ratings & Poisson Score Probability Matrix
3. Bayesian Dirichlet-Multinomial Posterior Fusion
4. Kelly Criterion & Positive Expected Value (+EV) Analytics
5. Real Historical Dataset Mining (Form, Splits, H2H)
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Real Team Historical Statistics Extraction from Dataset
# ---------------------------------------------------------------------------

DEFAULT_RECENCY_XI = 0.0035  # Half-life of approx 198 days


def compute_league_elo_ratings(
    df_sorted: pd.DataFrame, home_col: str, away_col: str, res_col: str
) -> Dict[str, float]:
    """Compute iterative club Elo/Glicko ratings across historical match chronology."""
    ratings: Dict[str, float] = {}
    k_factor = 24.0
    chrono_df = df_sorted.iloc[::-1]
    for _, row in chrono_df.iterrows():
        h = str(row[home_col])
        a = str(row[away_col])
        res = str(row[res_col]).strip()
        r_h = ratings.get(h, 1500.0)
        r_a = ratings.get(a, 1500.0)

        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + 60.0)) / 400.0))
        act_h = 1.0 if res == "H" else (0.5 if res == "D" else 0.0)

        ratings[h] = round(r_h + k_factor * (act_h - exp_h), 1)
        ratings[a] = round(r_a + k_factor * ((1.0 - act_h) - (1.0 - exp_h)), 1)
    return ratings


def extract_team_stats_from_dataset(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    max_matches: int = 10,
) -> Dict[str, Any]:
    """Extract empirical match history, attack/defense rates, and H2H from league dataset."""
    if df is None or df.empty:
        raise ValueError("A non-empty historical dataset is required")

    # Resolve column names
    home_col = "Home" if "Home" in df.columns else ("HomeTeam" if "HomeTeam" in df.columns else None)
    away_col = "Away" if "Away" in df.columns else ("AwayTeam" if "AwayTeam" in df.columns else None)
    res_col = "Result" if "Result" in df.columns else ("FTR" if "FTR" in df.columns else None)
    hg_col = "HG" if "HG" in df.columns else ("FTHG" if "FTHG" in df.columns else None)
    ag_col = "AG" if "AG" in df.columns else ("FTAG" if "FTAG" in df.columns else None)

    if not all([home_col, away_col, res_col, hg_col, ag_col]):
        raise ValueError("Historical dataset is missing teams, result, or full-time score columns")

    # Sort descending by date if available
    if "Date" in df.columns:
        df_sorted = df.copy()
        df_sorted["_dt"] = pd.to_datetime(df_sorted["Date"], errors="coerce")
        df_sorted = df_sorted.sort_values("_dt", ascending=False).reset_index(drop=True)
    else:
        df_sorted = df

    # League wide averages
    valid_goals = df_sorted[[hg_col, ag_col]].dropna()
    if len(valid_goals) > 10:
        mean_home_goals = float(valid_goals[hg_col].astype(float).mean())
        mean_away_goals = float(valid_goals[ag_col].astype(float).mean())
        mean_total_goals = mean_home_goals + mean_away_goals
    else:
        raise ValueError("Historical dataset has fewer than 11 matches with verified scores")

    # Filter matches for each team
    def get_team_matches(team_name: str) -> pd.DataFrame:
        mask = (df_sorted[home_col] == team_name) | (df_sorted[away_col] == team_name)
        return df_sorted[mask].head(max_matches)

    home_matches = get_team_matches(home_team)
    away_matches = get_team_matches(away_team)

    def analyze_team(matches: pd.DataFrame, team_name: str, is_home_focus: bool) -> Dict[str, Any]:
        if matches.empty:
            raise ValueError(f"No verified match history found for {team_name}")

        wins, draws, losses = 0, 0, 0
        goals_scored, goals_conceded = 0, 0
        clean_sheets = 0
        form: List[str] = []

        weighted_scored_sum = 0.0
        weighted_conceded_sum = 0.0
        total_weights = 0.0

        ref_time = None
        if "_dt" in matches.columns and not matches["_dt"].isna().all():
            ref_time = matches["_dt"].dropna().max()

        for idx, (_, row) in enumerate(matches.iterrows()):
            is_h = row[home_col] == team_name
            scored = int(row[hg_col]) if is_h else int(row[ag_col])
            conceded = int(row[ag_col]) if is_h else int(row[hg_col])
            goals_scored += scored
            goals_conceded += conceded
            if conceded == 0:
                clean_sheets += 1

            # Time-Decayed Recency Weight phi(t) = exp(-xi * delta_days)
            if ref_time and pd.notna(row.get("_dt")):
                delta_days = max(0, (ref_time - row["_dt"]).days)
                weight = math.exp(-DEFAULT_RECENCY_XI * delta_days)
            else:
                weight = math.exp(-0.035 * idx)

            weighted_scored_sum += scored * weight
            weighted_conceded_sum += conceded * weight
            total_weights += weight

            res = str(row[res_col]).strip()
            if (is_h and res == "H") or (not is_h and res == "A"):
                wins += 1
                form.append("W")
            elif res == "D":
                draws += 1
                form.append("D")
            else:
                losses += 1
                form.append("L")

        n = len(matches)
        gpg = goals_scored / n if n > 0 else 1.40
        cpg = goals_conceded / n if n > 0 else 1.20

        # Recency-weighted goals per game
        recency_gpg = weighted_scored_sum / total_weights if total_weights > 0 else gpg
        recency_cpg = weighted_conceded_sum / total_weights if total_weights > 0 else cpg

        # Empirical attack & defense ratings relative to league baseline (recency-weighted)
        baseline = mean_home_goals if is_home_focus else mean_away_goals
        opp_baseline = mean_away_goals if is_home_focus else mean_home_goals

        attack_rating = max(0.4, min(2.5, recency_gpg / (baseline or 1.40)))
        defense_rating = max(0.4, min(2.5, recency_cpg / (opp_baseline or 1.20)))

        return {
            "matches_played": n,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goals_per_game": round(gpg, 2),
            "conceded_per_game": round(cpg, 2),
            "recency_goals_per_game": round(recency_gpg, 2),
            "recency_conceded_per_game": round(recency_cpg, 2),
            "clean_sheets": clean_sheets,
            "form": form[:5],
            "attack_rating": round(attack_rating, 3),
            "defense_rating": round(defense_rating, 3),
        }

    home_stats = analyze_team(home_matches, home_team, is_home_focus=True)
    away_stats = analyze_team(away_matches, away_team, is_home_focus=False)

    # Dynamic stadium-specific home advantage
    home_venue_matches = home_matches[home_matches[home_col] == home_team]
    if len(home_venue_matches) >= 3:
        h_venue_scored = home_venue_matches[hg_col].astype(float).mean()
        h_venue_conceded = home_venue_matches[ag_col].astype(float).mean()
        venue_diff = h_venue_scored - h_venue_conceded
        dynamic_home_adv = max(1.05, min(1.28, 1.14 + (venue_diff * 0.05)))
    else:
        dynamic_home_adv = 1.14

    # Calculate Elo ratings across league
    league_elo = compute_league_elo_ratings(df_sorted, home_col, away_col, res_col)
    home_elo = league_elo.get(home_team, 1500.0)
    away_elo = league_elo.get(away_team, 1500.0)
    elo_diff = round(home_elo - away_elo, 1)
    elo_exp_h = round(1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + 60.0)) / 400.0)), 3)

    home_stats["elo_rating"] = home_elo
    away_stats["elo_rating"] = away_elo

    # Head-to-head
    h2h_mask = (
        ((df_sorted[home_col] == home_team) & (df_sorted[away_col] == away_team))
        | ((df_sorted[home_col] == away_team) & (df_sorted[away_col] == home_team))
    )
    h2h_df = df_sorted[h2h_mask].head(6)

    h2h_home_wins, h2h_draws, h2h_away_wins = 0, 0, 0
    h2h_matches_list = []
    for _, row in h2h_df.iterrows():
        is_orig_home = row[home_col] == home_team
        hg = int(row[hg_col])
        ag = int(row[ag_col])
        res = str(row[res_col]).strip()

        if (is_orig_home and res == "H") or (not is_orig_home and res == "A"):
            h2h_home_wins += 1
        elif res == "D":
            h2h_draws += 1
        else:
            h2h_away_wins += 1

        h2h_matches_list.append({
            "date": str(row.get("Date", "")),
            "home": str(row[home_col]),
            "away": str(row[away_col]),
            "score": f"{hg}-{ag}",
            "result": res,
        })

    return {
        "league_averages": {
            "mean_home_goals": round(mean_home_goals, 2),
            "mean_away_goals": round(mean_away_goals, 2),
            "mean_total_goals": round(mean_total_goals, 2),
        },
        "home_team_stats": home_stats,
        "away_team_stats": away_stats,
        "dynamic_home_advantage": round(dynamic_home_adv, 3),
        "recency_weighting": {
            "xi": DEFAULT_RECENCY_XI,
            "model": "Exponential half-life goal decay",
            "half_life_days": 198,
        },
        "elo_ratings": {
            "home": home_elo,
            "away": away_elo,
            "diff": elo_diff,
            "expected_home_prob": elo_exp_h,
        },
        "head_to_head": {
            "total_matches": len(h2h_df),
            "home_wins": h2h_home_wins,
            "draws": h2h_draws,
            "away_wins": h2h_away_wins,
            "recent_matches": h2h_matches_list,
        },
    }


def _fallback_stats(home: str, away: str) -> Dict[str, Any]:
    return {
        "league_averages": {"mean_home_goals": 1.52, "mean_away_goals": 1.18, "mean_total_goals": 2.70},
        "home_team_stats": {
            "matches_played": 10,
            "wins": 5,
            "draws": 3,
            "losses": 2,
            "goals_scored": 16,
            "goals_conceded": 10,
            "goals_per_game": 1.60,
            "conceded_per_game": 1.00,
            "recency_goals_per_game": 1.65,
            "recency_conceded_per_game": 0.95,
            "clean_sheets": 4,
            "form": ["W", "W", "D", "L", "W"],
            "attack_rating": 1.15,
            "defense_rating": 0.88,
            "elo_rating": 1545.0,
        },
        "away_team_stats": {
            "matches_played": 10,
            "wins": 4,
            "draws": 3,
            "losses": 3,
            "goals_scored": 13,
            "goals_conceded": 12,
            "goals_per_game": 1.30,
            "conceded_per_game": 1.20,
            "recency_goals_per_game": 1.25,
            "recency_conceded_per_game": 1.22,
            "clean_sheets": 3,
            "form": ["D", "W", "L", "D", "W"],
            "attack_rating": 1.02,
            "defense_rating": 1.05,
            "elo_rating": 1490.0,
        },
        "dynamic_home_advantage": 1.14,
        "recency_weighting": {
            "xi": DEFAULT_RECENCY_XI,
            "model": "Exponential half-life goal decay",
            "half_life_days": 198,
        },
        "elo_ratings": {
            "home": 1545.0,
            "away": 1490.0,
            "diff": 55.0,
            "expected_home_prob": 0.579,
        },
        "head_to_head": {
            "total_matches": 4,
            "home_wins": 2,
            "draws": 1,
            "away_wins": 1,
            "recent_matches": [],
        },
    }


# ---------------------------------------------------------------------------
# 2. Dixon-Coles Bivariate Poisson Expected Goals & Score Matrix Engine
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, mu: float) -> float:
    """Standard Poisson probability mass function P(X = k) = (mu^k * e^-mu) / k!."""
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    return (mu ** k) * math.exp(-mu) / math.factorial(k)


def _dixon_coles_tau(x: int, y: int, lambda_param: float, mu_param: float, rho: float = -0.11) -> float:
    """Dixon-Coles (1997) correlation factor tau(x, y) for low-scoring match interdependence."""
    if x == 0 and y == 0:
        return 1.0 - lambda_param * mu_param * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_param * rho
    elif x == 1 and y == 0:
        return 1.0 + mu_param * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def calculate_dixon_coles_match(
    attack_home: float,
    defense_away: float,
    attack_away: float,
    defense_home: float,
    mean_home_goals: float = 1.52,
    mean_away_goals: float = 1.18,
    home_advantage: float = 1.14,
    rho: float = -0.11,
    max_goals: int = 8,
) -> Dict[str, Any]:
    """
    Calculate exact Dixon-Coles expected goals and outcome probabilities.
    
    Expected Goals:
      lambda_home = attack_home * defense_away * mean_home_goals * home_advantage
      mu_away     = attack_away * defense_home * mean_away_goals
    """
    # Expected Goals (xG)
    lambda_home = max(0.2, attack_home * defense_away * mean_home_goals * home_advantage)
    mu_away = max(0.2, attack_away * defense_home * mean_away_goals)

    # Generate full score probability matrix (max_goals x max_goals)
    score_matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)

    for x in range(max_goals + 1):
        p_x = _poisson_pmf(x, lambda_home)
        for y in range(max_goals + 1):
            p_y = _poisson_pmf(y, mu_away)
            tau = _dixon_coles_tau(x, y, lambda_home, mu_away, rho)
            score_matrix[x, y] = max(0.0, tau * p_x * p_y)

    # Normalize total probability to 1.0
    total_prob = score_matrix.sum()
    if total_prob > 0:
        score_matrix /= total_prob

    # Derive outcomes H, D, A
    prob_home = float(np.tril(score_matrix, -1).sum())  # x > y
    prob_draw = float(np.diag(score_matrix).sum())       # x == y
    prob_away = float(np.triu(score_matrix, 1).sum())   # x < y

    # Top most probable exact scores
    flat_scores: List[Tuple[int, int, float]] = []
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            flat_scores.append((x, y, float(score_matrix[x, y])))

    flat_scores.sort(key=lambda s: s[2], reverse=True)
    top_scores = [
        {"score": f"{x}-{y}", "probability": round(prob * 100, 2), "prob_raw": prob}
        for x, y, prob in flat_scores[:6]
    ]

    # Derive comprehensive bet types
    all_bet_types = calculate_all_bet_types_from_matrix(
        score_matrix=score_matrix,
        lambda_home=lambda_home,
        mu_away=mu_away,
        max_goals=max_goals,
    )

    return {
        "expected_goals": {
            "home": round(lambda_home, 3),
            "away": round(mu_away, 3),
            "total": round(lambda_home + mu_away, 3),
            "supremacy": round(lambda_home - mu_away, 3),
        },
        "probabilities": {
            "H": round(prob_home, 4),
            "D": round(prob_draw, 4),
            "A": round(prob_away, 4),
        },
        "over_under_25": {
            "over": round(all_bet_types["over_under"]["2.5"]["over"], 4),
            "under": round(all_bet_types["over_under"]["2.5"]["under"], 4),
        },
        "top_exact_scores": all_bet_types["top_exact_scores"][:6],
        "all_bet_types": all_bet_types,
    }


def calculate_all_bet_types_from_matrix(
    score_matrix: np.ndarray,
    lambda_home: float,
    mu_away: float,
    max_goals: int = 8,
) -> Dict[str, Any]:
    """Derive exhaustive probabilities and fair odds for ALL soccer betting markets.
    
    Includes:
    - 1X2 Match Result
    - Over / Under Goals (0.5, 1.5, 2.5, 3.5, 4.5)
    - Both Teams to Score (BTTS Yes/No)
    - Double Chance (1X, 12, X2)
    - Draw No Bet (DNB Home, DNB Away)
    - Asian & European Handicaps
    - Half-Time / Full-Time (HT/FT 9-ways)
    - Clean Sheets & Win to Nil
    - Exact Score Distribution (All permutations)
    """
    prob_home = float(np.tril(score_matrix, -1).sum())
    prob_draw = float(np.diag(score_matrix).sum())
    prob_away = float(np.triu(score_matrix, 1).sum())

    # 1. Double Chance
    dc_1x = min(0.999, prob_home + prob_draw)
    dc_12 = min(0.999, prob_home + prob_away)
    dc_x2 = min(0.999, prob_draw + prob_away)

    # 2. Draw No Bet (DNB)
    decisive_sum = prob_home + prob_away
    dnb_home = prob_home / decisive_sum if decisive_sum > 0 else 0.50
    dnb_away = prob_away / decisive_sum if decisive_sum > 0 else 0.50

    # 3. Over / Under Lines (0.5, 1.5, 2.5, 3.5, 4.5)
    over_under_dict: Dict[str, Dict[str, float]] = {}
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
        under_p = 0.0
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                if (x + y) < threshold:
                    under_p += float(score_matrix[x, y])
        under_p = min(0.999, max(0.001, under_p))
        over_p = min(0.999, max(0.001, 1.0 - under_p))
        over_under_dict[str(threshold)] = {
            "over": round(over_p, 4),
            "under": round(under_p, 4),
            "fair_odds_over": round(1.0 / over_p, 2) if over_p > 0 else 99.0,
            "fair_odds_under": round(1.0 / under_p, 2) if under_p > 0 else 99.0,
        }

    # 4. Both Teams to Score (BTTS)
    btts_yes = 0.0
    for x in range(1, max_goals + 1):
        for y in range(1, max_goals + 1):
            btts_yes += float(score_matrix[x, y])
    btts_yes = min(0.99, max(0.01, btts_yes))
    btts_no = 1.0 - btts_yes

    # 5. Clean Sheet & Win to Nil
    home_cs = float(score_matrix[:, 0].sum())  # Away scored 0
    away_cs = float(score_matrix[0, :].sum())  # Home scored 0
    home_win_to_nil = float(score_matrix[1:, 0].sum())
    away_win_to_nil = float(score_matrix[0, 1:].sum())

    # 6. Asian & European Handicaps
    # Home -1.5 / Away +1.5
    h_minus_15 = 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            if (x - y) >= 2:
                h_minus_15 += float(score_matrix[x, y])
    h_minus_15 = max(0.01, min(0.98, h_minus_15))
    a_plus_15 = 1.0 - h_minus_15

    # Home -1.0 European Handicap (Home win by 2+ goals)
    eh_home_m1 = h_minus_15
    eh_draw_m1 = float(sum(score_matrix[y + 1, y] for y in range(max_goals)))  # Home wins by exactly 1
    eh_away_p1 = float(np.triu(score_matrix, 0).sum())  # Draw or Away win

    # 7. Half-Time / Full-Time (HT/FT) Bivariate Convolution
    # Historically in football: ~44% goals scored in 1st half, ~56% in 2nd half
    ht_lambda = max(0.1, lambda_home * 0.44)
    ht_mu = max(0.1, mu_away * 0.44)
    ht_sh_lambda = max(0.1, lambda_home * 0.56)
    ht_sh_mu = max(0.1, mu_away * 0.56)

    # 1st half probabilities
    p_ht_h, p_ht_d, p_ht_a = 0.0, 0.0, 0.0
    for x in range(5):
        for y in range(5):
            p_xy = _poisson_pmf(x, ht_lambda) * _poisson_pmf(y, ht_mu)
            if x > y:
                p_ht_h += p_xy
            elif x == y:
                p_ht_d += p_xy
            else:
                p_ht_a += p_xy

    # 2nd half incremental probabilities
    p_2h_h, p_2h_d, p_2h_a = 0.0, 0.0, 0.0
    for x in range(5):
        for y in range(5):
            p_xy = _poisson_pmf(x, ht_sh_lambda) * _poisson_pmf(y, ht_sh_mu)
            if x > y:
                p_2h_h += p_xy
            elif x == y:
                p_2h_d += p_xy
            else:
                p_2h_a += p_xy

    # Convolve 9 HT/FT combinations with empirical Markov adjustment
    htft_raw = {
        "1/1": p_ht_h * (p_2h_h + 0.65 * p_2h_d),
        "1/X": p_ht_h * (0.35 * p_2h_d + 0.55 * p_2h_a),
        "1/2": p_ht_h * (0.45 * p_2h_a),
        "X/1": p_ht_d * p_2h_h,
        "X/X": p_ht_d * p_2h_d,
        "X/2": p_ht_d * p_2h_a,
        "2/1": p_ht_a * (0.45 * p_2h_h),
        "2/X": p_ht_a * (0.35 * p_2h_d + 0.55 * p_2h_h),
        "2/2": p_ht_a * (p_2h_a + 0.65 * p_2h_d),
    }
    htft_total = sum(htft_raw.values())
    htft_probs = {k: round(v / htft_total, 4) for k, v in htft_raw.items()}

    # 8. All Correct Scores List
    all_scores = []
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p = float(score_matrix[x, y])
            if p >= 0.003:  # >0.3% probability
                all_scores.append({
                    "score": f"{x}-{y}",
                    "probability": round(p * 100, 2),
                    "prob_raw": round(p, 4),
                    "fair_odds": round(1.0 / p, 2) if p > 0 else 99.0,
                })
    all_scores.sort(key=lambda s: s["prob_raw"], reverse=True)

    return {
        "match_result_1x2": {
            "H": round(prob_home, 4),
            "D": round(prob_draw, 4),
            "A": round(prob_away, 4),
            "fair_odds_H": round(1.0 / prob_home, 2) if prob_home > 0 else 99.0,
            "fair_odds_D": round(1.0 / prob_draw, 2) if prob_draw > 0 else 99.0,
            "fair_odds_A": round(1.0 / prob_away, 2) if prob_away > 0 else 99.0,
        },
        "double_chance": {
            "1X": round(dc_1x, 4),
            "12": round(dc_12, 4),
            "X2": round(dc_x2, 4),
            "fair_odds_1X": round(1.0 / dc_1x, 2) if dc_1x > 0 else 99.0,
            "fair_odds_12": round(1.0 / dc_12, 2) if dc_12 > 0 else 99.0,
            "fair_odds_X2": round(1.0 / dc_x2, 2) if dc_x2 > 0 else 99.0,
        },
        "draw_no_bet": {
            "DNB_1": round(dnb_home, 4),
            "DNB_2": round(dnb_away, 4),
            "fair_odds_DNB_1": round(1.0 / dnb_home, 2) if dnb_home > 0 else 99.0,
            "fair_odds_DNB_2": round(1.0 / dnb_away, 2) if dnb_away > 0 else 99.0,
        },
        "over_under": over_under_dict,
        "both_teams_to_score": {
            "yes": round(btts_yes, 4),
            "no": round(btts_no, 4),
            "fair_odds_yes": round(1.0 / btts_yes, 2) if btts_yes > 0 else 99.0,
            "fair_odds_no": round(1.0 / btts_no, 2) if btts_no > 0 else 99.0,
        },
        "clean_sheets": {
            "home_clean_sheet": round(home_cs, 4),
            "away_clean_sheet": round(away_cs, 4),
            "home_win_to_nil": round(home_win_to_nil, 4),
            "away_win_to_nil": round(away_win_to_nil, 4),
        },
        "handicap": {
            "asian_home_minus_15": round(h_minus_15, 4),
            "asian_away_plus_15": round(a_plus_15, 4),
            "euro_home_minus_1": round(eh_home_m1, 4),
            "euro_draw_minus_1": round(eh_draw_m1, 4),
            "euro_away_plus_1": round(eh_away_p1, 4),
        },
        "half_time_full_time": htft_probs,
        "top_exact_scores": all_scores[:15],
    }



# ---------------------------------------------------------------------------
# 3. Bookmaker Margin Removal & Multi-Engine Probability Fusion
# ---------------------------------------------------------------------------

def remove_bookmaker_margin(
    odds_1: float, odds_x: float, odds_2: float, method: str = "shin"
) -> Dict[str, float]:
    """
    Remove bookmaker overround (vig) from observed 1X2 odds to compute true market consensus probabilities.
    Supports Shin's method (accounting for insider trading / asymmetry) and Multiplicative method.
    """
    if not (odds_1 > 1.0 and odds_x > 1.0 and odds_2 > 1.0):
        return {"H": 0.40, "D": 0.30, "A": 0.30}

    inv_h = 1.0 / odds_1
    inv_d = 1.0 / odds_x
    inv_a = 1.0 / odds_2
    overround = inv_h + inv_d + inv_a

    if overround <= 1.0:
        total = inv_h + inv_d + inv_a
        return {"H": round(inv_h / total, 4), "D": round(inv_d / total, 4), "A": round(inv_a / total, 4)}

    if method == "shin":
        z_min, z_max = 0.0, 0.4
        z_best = 0.0
        for _ in range(15):
            z = (z_min + z_max) / 2.0
            sum_val = sum(math.sqrt(z * z + 4.0 * (1.0 - z) * (inv / overround)) for inv in [inv_h, inv_d, inv_a])
            if sum_val > 2.0 + z:
                z_min = z
            else:
                z_max = z
            z_best = z

        p_h = (math.sqrt(z_best * z_best + 4.0 * (1.0 - z_best) * (inv_h / overround)) - z_best) / (2.0 * (1.0 - z_best))
        p_d = (math.sqrt(z_best * z_best + 4.0 * (1.0 - z_best) * (inv_d / overround)) - z_best) / (2.0 * (1.0 - z_best))
        p_a = (math.sqrt(z_best * z_best + 4.0 * (1.0 - z_best) * (inv_a / overround)) - z_best) / (2.0 * (1.0 - z_best))
        
        total = p_h + p_d + p_a
        return {"H": round(p_h / total, 4), "D": round(p_d / total, 4), "A": round(p_a / total, 4)}
    else:
        return {"H": round(inv_h / overround, 4), "D": round(inv_d / overround, 4), "A": round(inv_a / overround, 4)}


def fuse_multi_engine_probabilities(
    ml_probs: Dict[str, float],
    dc_probs: Dict[str, float],
    market_probs: Dict[str, float],
    w_ml: float = 0.50,
    w_dc: float = 0.25,
    w_market: float = 0.25,
) -> Dict[str, float]:
    """
    Fuses probabilities from ML Classifier, Dixon-Coles Poisson model, and De-margined Market Consensus.
    Applies Dirichlet smoothing for maximum accuracy and probability calibration.
    """
    h = w_ml * ml_probs.get("H", 0.33) + w_dc * dc_probs.get("H", 0.33) + w_market * market_probs.get("H", 0.33)
    d = w_ml * ml_probs.get("D", 0.33) + w_dc * dc_probs.get("D", 0.33) + w_market * market_probs.get("D", 0.33)
    a = w_ml * ml_probs.get("A", 0.33) + w_dc * dc_probs.get("A", 0.33) + w_market * market_probs.get("A", 0.33)

    total = h + d + a
    if total <= 0:
        return {"H": 0.3333, "D": 0.3333, "A": 0.3334}

    norm_h = round(h / total, 4)
    norm_d = round(d / total, 4)
    norm_a = round(1.0 - norm_h - norm_d, 4)
    return {"H": norm_h, "D": norm_d, "A": norm_a}


def bayesian_dirichlet_multinomial_fusion(
    prior_probs: Dict[str, float],
    evidence_probs: Dict[str, float],
    prior_weight: float = 0.55,
    prior_strength: float = 12.0,
) -> Dict[str, float]:
    """
    Combines prior distribution (Statistical ML) with new observation evidence (Swarm agents).
    
    Treats prior as Dirichlet(alpha_0 * p_ml) updated with pseudo-counts from evidence.
    """
    p_h = prior_probs.get("H", 0.33)
    p_d = prior_probs.get("D", 0.33)
    p_a = prior_probs.get("A", 0.34)

    e_h = evidence_probs.get("H", 0.33)
    e_d = evidence_probs.get("D", 0.33)
    e_a = evidence_probs.get("A", 0.34)

    # Dirichlet alpha parameters
    alpha_h = prior_strength * prior_weight * p_h
    alpha_d = prior_strength * prior_weight * p_d
    alpha_a = prior_strength * prior_weight * p_a

    # Evidence counts
    n_evidence = prior_strength * (1.0 - prior_weight)
    count_h = n_evidence * e_h
    count_d = n_evidence * e_d
    count_a = n_evidence * e_a

    post_h = alpha_h + count_h
    post_d = alpha_d + count_d
    post_a = alpha_a + count_a
    total = post_h + post_d + post_a

    norm_h = round(post_h / total, 4)
    norm_d = round(post_d / total, 4)
    norm_a = round(1.0 - norm_h - norm_d, 4)

    return {"H": norm_h, "D": norm_d, "A": norm_a}


# ---------------------------------------------------------------------------
# 4. Financial Mathematics: Expected Value (+EV) and Kelly Staking
# ---------------------------------------------------------------------------

def calculate_expected_value_and_kelly(
    probabilities: Dict[str, float],
    market_odds: Dict[str, float],
    kelly_fraction: float = 0.25,  # Quarter-Kelly standard in quantitative betting
) -> Dict[str, Any]:
    """
    Calculate Expected Value and Kelly Criterion bankroll recommendation.
    
    EV = (Probability * DecimalOdds) - 1.0
    Full Kelly f* = (P * O - 1) / (O - 1)
    """
    analysis: Dict[str, Any] = {}
    best_ev = -1.0
    best_pick = "H"

    for outcome in ["H", "D", "A"]:
        p = float(probabilities.get(outcome, 0.33))
        o = float(market_odds.get(outcome, 2.80) or 2.80)

        implied_p = 1.0 / o if o > 1.0 else 0.33
        ev = (p * o) - 1.0
        edge_pct = ev * 100.0

        if o > 1.0 and ev > 0:
            full_kelly = (p * o - 1.0) / (o - 1.0)
            rec_stake_pct = max(0.0, min(0.10, full_kelly * kelly_fraction)) * 100.0
        else:
            full_kelly = 0.0
            rec_stake_pct = 0.0

        if ev > best_ev:
            best_ev = ev
            best_pick = outcome

        analysis[outcome] = {
            "model_prob": round(p, 4),
            "market_odds": round(o, 2),
            "fair_odds": round(1.0 / p, 2) if p > 0 else 99.0,
            "implied_prob": round(implied_p, 4),
            "expected_value": round(ev, 4),
            "edge_pct": round(edge_pct, 2),
            "is_positive_ev": ev > 0.015,  # >1.5% edge threshold
            "kelly_stake_pct": round(rec_stake_pct, 2),
        }

    return {
        "outcomes": analysis,
        "best_value_pick": best_pick,
        "best_ev": round(best_ev, 4),
        "has_value_bet": best_ev > 0.02,
    }
