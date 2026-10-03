import logging
import uuid
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd

from backend.app.config import setup_ml_path
from backend.app.services.model_artifact import serialize_model

logger = logging.getLogger(__name__)


def _classification_ctor_kwargs(tt, hyperparams: Optional[dict]) -> dict[str, Any]:
    """ProphitBet classifiers require league_id, model_id, target_type, calibrate_probabilities."""
    base = {
        "league_id": "saas",
        "model_id": str(uuid.uuid4()),
        "target_type": tt,
        "calibrate_probabilities": False,
    }
    merged = {**base, **(hyperparams or {})}
    merged["target_type"] = tt
    return merged


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip non-JSON types (e.g. Enum) for DB / API responses."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, Enum):
            out[k] = v.value
        else:
            out[k] = v
    return out

MODEL_CLASS_MAP = {
    "logistic": "LogisticRegressor",
    "decision_tree": "DecisionTree",
    "naive_bayes": "NaiveBayes",
    "knn": "KNN",
    "svm": "SVM",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "discriminant": "DiscriminantAnalysisClassifier",
}


def prepare_target_data(df, target_type):
    from src.preprocessing.utils.target import parse_target, valid_target_rows, target_columns
    target = parse_target(target_type)
    if 'ScoreScope' in df.columns:
        df = df[df['ScoreScope'] == 'regulation']
    df = df.loc[valid_target_rows(df, target)].copy()
    # Discard unrelated raw outcomes, never substitute zero for an unknown label.
    raw = {'Result', 'HG', 'AG', 'HC', 'AC', 'HST', 'AST', 'Result-U/O'}
    keep = set(target_columns(target))
    if target.value == 'result':
        keep.update(('HG', 'AG'))
    return df.drop(columns=list(raw - keep), errors='ignore')


def get_model_class(model_type: str):
    setup_ml_path()
    from src.models.classifiers import (
        LogisticRegressor,
        DecisionTree,
        NaiveBayes,
        KNN,
        SVM,
        RandomForest,
        XGBoost,
        DiscriminantAnalysisClassifier,
    )
    mapping = {
        "logistic": LogisticRegressor,
        "decision_tree": DecisionTree,
        "naive_bayes": NaiveBayes,
        "knn": KNN,
        "svm": SVM,
        "random_forest": RandomForest,
        "xgboost": XGBoost,
        "discriminant": DiscriminantAnalysisClassifier,
    }
    return mapping.get(model_type)


def train_model(
    df: pd.DataFrame,
    model_type: str,
    target_type: str = "result",
    hyperparams: Optional[dict] = None,
    use_odds: bool = True,
) -> dict:
    """
    Train a model on the given dataframe.
    Returns dict with 'model_bytes', 'metrics', 'hyperparams'.
    """
    setup_ml_path()
    from src.models.temporal_evaluation import evaluate_temporally
    from src.preprocessing.utils.target import parse_target

    from backend.app.services.dataset_validation import validate_dataset
    validate_dataset(df)
    # Historical closing prices without an observation timestamp cannot be
    # proven to have existed before kickoff. Do not train on them by default.
    odds_requested = use_odds
    odds_timing_verified = False
    if use_odds and 'OddsObservedAt' in df.columns:
        observed = pd.to_datetime(df['OddsObservedAt'], errors='coerce', utc=True)
        kickoff = pd.to_datetime(df['Date'], errors='coerce', utc=True)
        odds_timing_verified = bool((observed.notna() & kickoff.notna() & (observed < kickoff)).all())
    use_odds = use_odds and odds_timing_verified
    df = df.drop(columns=['OddsObservedAt'], errors='ignore')
    df = prepare_target_data(df, target_type)
    if parse_target(target_type).value != "result":
        df = df.drop(columns=["1", "X", "2"], errors="ignore")

    if "ScoreScope" in df.columns:
        df = df[df["ScoreScope"] == "regulation"]
    df = df.drop(columns=["Source", "ScoreScope"], errors="ignore")
    use_odds = use_odds and parse_target(target_type).value == "result" and all(col in df.columns for col in ("1", "X", "2"))
    if not use_odds:
        df = df.drop(columns=["1", "X", "2"], errors="ignore")
    df = df.sort_values("Date", ascending=False) if "Date" in df.columns else df
    df = df.dropna(how="any").reset_index(drop=True)
    if len(df) < 50:
        raise ValueError(f"Insufficient rows after dropping NaNs ({len(df)}); need at least 50.")

    model_cls = get_model_class(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}")

    tt = parse_target(target_type)
    params = _classification_ctor_kwargs(tt, hyperparams)
    model = model_cls(**params)
    evaluation_report = evaluate_temporally(df, lambda: model_cls(**params), tt)
    model = model_cls(**params)
    model.fit(train_df=df, eval_df=None)
    model._input_columns = list(df.columns)
    model._requires_odds = use_odds

    model_bytes = serialize_model(model)

    import hashlib
    dataset_fingerprint = hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()
    final = evaluation_report['folds'][-1]
    metrics = {"evaluation": "chronological_final_holdout", "uses_odds": use_odds,
               "target_type": tt.value, "training_rows": len(df),
               "dataset_sha256": dataset_fingerprint,
               "odds_excluded_unverified_timing": bool(odds_requested and not odds_timing_verified),
               "temporal_report": evaluation_report,
               "beats_frequency_baseline": final['model']['log_loss'] < final['baseline']['log_loss']}
    for col in ("Accuracy", "Precision", "Recall", "F1"):
        metrics[col] = round(final['model'][col.lower()], 4)
    metrics['LogLoss'] = final['model']['log_loss']
    metrics['Brier'] = final['model']['brier']

    return {
        "model_bytes": model_bytes,
        "metrics": metrics,
        "hyperparams": _json_safe_params(params),
    }


def auto_tune_model(
    df: pd.DataFrame,
    model_type: str,
    target_type: str = "result",
    n_trials: int = 50,
    fixed_params: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Run Optuna auto-tuning, then train the best model.
    Returns dict with 'model_bytes', 'metrics', 'hyperparams', 'best_trial'.
    """
    setup_ml_path()
    from src.models.tuner import Tuner
    from src.preprocessing.utils.target import parse_target

    df = prepare_target_data(df, target_type)
    if parse_target(target_type).value != "result":
        df = df.drop(columns=["1", "X", "2"], errors="ignore")

    if "ScoreScope" in df.columns:
        df = df[df["ScoreScope"] == "regulation"]
    df = df.drop(columns=["Source", "ScoreScope"], errors="ignore")
    if "Date" in df.columns:
        df = df.sort_values("Date", ascending=False)
    df = df.dropna().reset_index(drop=True)
    if len(df) < 50:
        raise ValueError("Insufficient complete rows for tuning; need at least 50.")

    model_cls = get_model_class(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}")

    tt = parse_target(target_type)

    fixed = _classification_ctor_kwargs(tt, fixed_params)
    tunable_params: dict[str, Any] = {}
    for param_name in (
        "n_estimators",
        "criterion",
        "min_samples_leaf",
        "min_samples_split",
        "max_features",
        "max_depth",
        "class_weight",
        "penalty",
        "n_neighbors",
        "weights",
        "kernel",
        "C",
        "gamma",
    ):
        try:
            tunable_params[param_name] = model_cls.get_suggest_param_values(param=param_name)
        except ValueError:
            continue
    if not tunable_params:
        raise ValueError(f"No tunable parameters discovered for model type {model_type!r}")

    tuner = Tuner(
        model_cls=model_cls,
        fixed_params=fixed,
        tunable_params=tunable_params,
        df=df,
        metric="Accuracy",
    )
    study = tuner.tune(trials=n_trials, progress_callback=progress_callback)

    best_params = {**fixed, **study.best_trial.params}
    result = train_model(df=df, model_type=model_type, target_type=target_type, hyperparams=best_params)
    result["best_trial"] = {
        "number": study.best_trial.number,
        "value": round(study.best_trial.value, 4),
        "params": study.best_trial.params,
    }
    result["hyperparams"] = _json_safe_params(best_params)
    return result
