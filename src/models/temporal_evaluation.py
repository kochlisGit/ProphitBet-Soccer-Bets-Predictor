"""Expanding-window probability evaluation with an untouched final date block."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from src.preprocessing.utils.target import construct_targets


def probability_scores(y, probabilities):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y, dtype=int)
    if (p.ndim != 2 or len(p) != len(y) or not len(y)
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-6)
            or (y < 0).any() or (y >= p.shape[1]).any()):
        raise ValueError('Invalid probabilities or evaluation labels')
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    bins = []
    for lower in np.arange(0, 1, .1):
        mask = (confidence >= lower) & (confidence < lower + .1 if lower < .9 else confidence <= 1)
        if mask.any():
            bins.append(dict(count=int(mask.sum()), confidence=float(confidence[mask].mean()),
                             accuracy=float(correct[mask].mean())))
    return dict(samples=len(y), accuracy=float(correct.mean()),
                precision=float(precision_score(y, p.argmax(axis=1), average='macro', zero_division=0)),
                recall=float(recall_score(y, p.argmax(axis=1), average='macro', zero_division=0)),
                f1=float(f1_score(y, p.argmax(axis=1), average='macro', zero_division=0)),
                log_loss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
                brier=float(((p - np.eye(p.shape[1])[y]) ** 2).sum(axis=1).mean()),
                calibration=bins)


def evaluate_temporally(df, model_factory, target_type, folds=3):
    dates = pd.to_datetime(df['Date'], errors='raise', utc=True).dt.normalize()
    if dates.isna().any():
        raise ValueError('Evaluation requires valid match dates')
    unique = np.sort(dates.unique())
    if len(unique) < folds + 2:
        raise ValueError('Not enough distinct match dates for temporal evaluation')
    blocks = np.array_split(unique, folds + 2)
    reports = []
    for index in range(1, len(blocks)):
        train = df.loc[dates < blocks[index][0]].copy()
        test = df.loc[dates.isin(blocks[index])].copy()
        model = model_factory()  # Fresh preprocessing and estimator for every fold.
        model.fit(train_df=train, eval_df=None)
        y = construct_targets(test, target_type)
        train_y = construct_targets(train, target_type)
        n_classes = 3 if getattr(target_type, 'value', target_type) == 'result' else 2
        raw = model.predict_proba(test)
        classes = np.asarray(model.classifier.classes_, dtype=int)
        probabilities = np.zeros((len(test), n_classes))
        probabilities[:, classes] = raw
        frequencies = (np.bincount(train_y, minlength=n_classes) + 1) / (len(train_y) + n_classes)
        reports.append(dict(
            kind='final_holdout' if index == len(blocks) - 1 else 'validation',
            training_rows=len(train), training_end=str(dates.loc[train.index].max()),
            evaluation_start=str(dates.loc[test.index].min()), evaluation_end=str(dates.loc[test.index].max()),
            model=probability_scores(y, probabilities),
            baseline=probability_scores(y, np.tile(frequencies, (len(test), 1))),
        ))
    return dict(method='expanding_date_blocks', folds=reports,
                limitations=['Historical feature availability must be audited separately.',
                             'Calibration uses top-label confidence; Brier is multiclass sum, not class average.'])
