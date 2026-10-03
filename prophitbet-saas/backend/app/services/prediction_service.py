import difflib
import logging

import pandas as pd

from backend.app.config import setup_ml_path
from backend.app.services.model_artifact import deserialize_model
from backend.app.services.team_mapping import normalize_team_name

logger = logging.getLogger(__name__)

def _coerce_team_to_column(raw: str, column: pd.Series) -> str | None:
    """
    Map a scraped fixture team name to a value that actually appears in `column`.
    construct_inputs_by_teams requires an exact row match; returning a non-existent
    label causes 'single positional indexer is out-of-bounds'.
    
    Now uses the centralized team_mapping module for better normalization.
    """
    name = (raw or "").strip()
    if not name:
        return None
    uniq = sorted({str(u).strip() for u in column.dropna().unique() if str(u).strip()}, key=len, reverse=True)
    if not uniq:
        return None
    
    # Use the new centralized team mapping
    mapped = normalize_team_name(name, uniq)
    if mapped:
        return mapped
    
    # Fallback to original logic if mapping fails
    if name in uniq:
        return name
    close = difflib.get_close_matches(name, uniq, n=1, cutoff=0.6)
    if close:
        return close[0]
    nl = name.lower()
    for u in uniq:
        ul = u.lower()
        if nl == ul or nl in ul or ul in nl:
            return u
    return None


def generate_predictions_for_league(
    model_bytes: bytes,
    league_df: pd.DataFrame,
    fixtures: list[dict],
    target_type: str = "result",
) -> list[dict]:
    setup_ml_path()
    from src.preprocessing.utils.inputs import construct_inputs_by_teams
    from src.preprocessing.utils.target import TargetType, parse_target, target_labels

    model = deserialize_model(model_bytes)
    requires_odds = getattr(model, "_requires_odds", True)
    tt = parse_target(target_type)
    if parse_target(model.target_type) != tt:
        raise ValueError("Requested market does not match the trained model target")
    results: list[dict] = []

    if league_df is None or league_df.empty or "Date" not in league_df.columns:
        logger.warning("generate_predictions_for_league: invalid league_df")
        return results

    columns = getattr(model, "_input_columns", None)
    if "ScoreScope" in league_df.columns:
        league_df = league_df[league_df["ScoreScope"] == "regulation"]
    if columns:
        missing = set(columns) - set(league_df.columns)
        if missing:
            raise ValueError(f"Model input columns missing: {sorted(missing)}")
        league_df = league_df[columns]
    df = league_df.sort_values("Date", ascending=False).reset_index(drop=True)

    input_parts: list[pd.DataFrame] = []
    raw_pairs: list[tuple[str, str, str | None, float | None, float | None, float | None]] = []
    for fixture in fixtures:
        if requires_odds and any(fixture.get(key) is None for key in ("odds_1", "odds_x", "odds_2")):
            logger.warning("skip fixture without observed 1X2 odds: %s vs %s", fixture.get("home_team"), fixture.get("away_team"))
            continue
        home_raw = fixture["home_team"]
        away_raw = fixture["away_team"]
        home = _coerce_team_to_column(home_raw, df["Home"])
        away = _coerce_team_to_column(away_raw, df["Away"])
        if home is None or away is None:
            continue
        match_row = pd.DataFrame(
            [
                {
                    "Home": home,
                    "Away": away,
                    **({"1": fixture["odds_1"], "X": fixture["odds_x"], "2": fixture["odds_2"]} if requires_odds else {}),
                }
            ]
        )
        try:
            input_parts.append(construct_inputs_by_teams(df=df, match_df=match_row, require_odds=requires_odds))
            raw_pairs.append((
                home_raw,
                away_raw,
                fixture.get("fixture_id"),
                fixture.get("odds_1"),
                fixture.get("odds_x"),
                fixture.get("odds_2"),
            ))
        except Exception as e:
            logger.warning("skip fixture %s vs %s: %s", home_raw, away_raw, e)
            continue

    if not input_parts:
        logger.warning("generate_predictions_for_league: no fixtures matched league team names")
        return results

    input_df = pd.concat(input_parts, axis=0, ignore_index=True)

    try:
        y_pred, _ = model.predict(input_df)
        proba = model.predict_proba(input_df)
    except Exception as e:
        logger.error("model.predict failed: %s", e)
        return results

    if tt == TargetType.RESULT:
        labels = ["H", "D", "A"]
        nlab = 3
    else:
        labels = target_labels(tt)
        nlab = 2

    from backend.app.services.math_engine import (
        remove_bookmaker_margin,
        fuse_multi_engine_probabilities,
        extract_team_stats_from_dataset,
        calculate_dixon_coles_match,
    )

    classes = model.classifier.classes_
    for i, (home_raw, away_raw, fixture_id, o1, ox, o2) in enumerate(raw_pairs):
        try:
            ml_prob_dict = {
                labels[int(classes[j])]: float(proba[i][j])
                for j in range(min(nlab, proba.shape[1]))
            }

            if tt == TargetType.RESULT:
                # 1. Market De-margined Probabilities (Shin's method)
                if o1 and ox and o2 and float(o1) > 1.0:
                    mkt_probs = remove_bookmaker_margin(float(o1), float(ox), float(o2), method="shin")
                else:
                    mkt_probs = ml_prob_dict.copy()

                # 2. Dixon-Coles Poisson xG Model
                dc_probs = ml_prob_dict.copy()
                try:
                    home_coerced = _coerce_team_to_column(home_raw, df["Home"])
                    away_coerced = _coerce_team_to_column(away_raw, df["Away"])
                    if home_coerced and away_coerced:
                        stats = extract_team_stats_from_dataset(df, home_coerced, away_coerced)
                        h_stats = stats["home_team_stats"]
                        a_stats = stats["away_team_stats"]
                        dc_res = calculate_dixon_coles_match(
                            attack_home=h_stats["attack_rating"],
                            defense_away=a_stats["defense_rating"],
                            attack_away=a_stats["attack_rating"],
                            defense_home=h_stats["defense_rating"],
                            mean_home_goals=stats["league_averages"]["mean_home_goals"],
                            mean_away_goals=stats["league_averages"]["mean_away_goals"],
                            home_advantage=stats["dynamic_home_advantage"],
                        )
                        dc_probs = dc_res["probabilities"]
                except Exception as dc_err:
                    logger.debug("Dixon-Coles estimation fallback for %s vs %s: %s", home_raw, away_raw, dc_err)

                # 3. Multi-Engine Bayesian Dirichlet Fusion
                fused_probs = fuse_multi_engine_probabilities(
                    ml_probs=ml_prob_dict,
                    dc_probs=dc_probs,
                    market_probs=mkt_probs,
                    w_ml=0.50,
                    w_dc=0.25,
                    w_market=0.25,
                )

                # Select highest probability prediction
                predicted = max(fused_probs, key=fused_probs.get)
                prob_dict = fused_probs
            else:
                predicted = labels[int(y_pred[i])]
                prob_dict = {k: round(v, 4) for k, v in ml_prob_dict.items()}

        except (IndexError, ValueError) as e:
            logger.warning("prediction row %s failed: %s", i, e)
            continue

        results.append(
            {
                "home_team": home_raw,
                "fixture_id": fixture_id,
                "away_team": away_raw,
                "predicted_result": predicted,
                "probabilities": prob_dict,
                "target_type": tt.value,
            }
        )

    return results
