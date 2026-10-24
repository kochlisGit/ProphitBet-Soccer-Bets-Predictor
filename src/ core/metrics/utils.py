import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score
)
from src.core import metrics
from src.core.preprocessing.utils.targets import TargetType

ODDS_COLUMNS = ['1', 'X', '2']
DOUBLE_CHANCE_ODD_COLUMNS = ['1X', '12', 'X2']
CLASSIFICATION_METRICS = ['Correct', 'Accuracy', 'F1', 'Precision', 'Recall']
PROBABILITY_METRICS = ['Log Loss', 'Brier Score', 'RPS', *[f'MIP-{odd}' for odd in ODDS_COLUMNS]]
SKILL_METRICS = [f'{prob_col} Skill' for prob_col in PROBABILITY_METRICS if 'MIP' not in prob_col] + ['MAPD']


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """ Computes the number of correct predictions and classification scores.

        Args
        ----
        :param y_true: true labels of inputs.
        :param y_pred: classifier's predicted labels.
    """

    report = pd.DataFrame({
        'Correct': [int(np.count_nonzero(y_true == y_pred))],
        'Accuracy': [accuracy_score(y_true=y_true, y_pred=y_pred)],
        'F1': [f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0.0)],
        'Precision': [precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0.0)],
        'Recall': [recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0.0)],
    })
    return report


def probabilistic_report(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """ Computes classification metrics, including Brier,

        Args
        ----
        :param y_true: true labels of inputs.
        :param y_prob: classifier's predicted probabilities for each label.
                       Its shape is expected to be (n_samples, n_classes).
    """

    labels = np.arange(y_prob.shape[1])
    report = pd.DataFrame({
        'Log Loss': [log_loss(y_true=y_true, y_proba=y_prob, labels=labels)],
        'Brier Score': [brier_score_loss(y_true=y_true, y_proba=y_prob, labels=labels)],
        'RPS': [metrics.ranked_probabilities_score(y_true=y_true, y_proba=y_prob)]
    })

    if y_prob.shape[1] == 2:
        report.insert(loc=3, column='AUC', value=roc_auc_score(y_true=y_true, y_score=y_prob[:, 1], labels=labels))
    else:
        if np.unique(y_true).shape[0] != labels.shape[0]:
            report.insert(loc=3, column='AUC', value=np.nan)
        else:
            report.insert(loc=3, column='AUC', value=roc_auc_score(y_true=y_true, y_score=y_prob, multi_class='ovo', labels=labels))

    return report


def skill_report(
        y_prob: np.ndarray,
        estimator_prob_report: pd.DataFrame,
        baseline_prob: np.ndarray,
        baseline_prob_report: pd.DataFrame
) -> pd.DataFrame:
    """ Computes skill metrics of the classifier using the baseline model (of the bookmaker).
        Its shape is expected to be (n_samples, n_classes).

        Args
        ----
        :param y_prob: (np.ndarray) classifier's predicted probabilities for each label.
        :param estimator_prob_report: (pd.DataFrame) dataframe of classifier's probabilistic metrics.
        :param baseline_prob: (np.ndarray) baseline model's predicted probabilities for each label.
        :param baseline_prob_report: (pd.DataFrame) dataframe of baseline's probabilistic metrics.
    """

    estimator_prob_report = estimator_prob_report.loc[:, [
        column
        for column in estimator_prob_report.columns
        if 'MIP' not in column
    ]]
    baseline_prob_report = baseline_prob_report.loc[:, [
        column
        for column in baseline_prob_report.columns
        if 'MIP' not in column
    ]]
    report = 1 - estimator_prob_report / baseline_prob_report
    report = report.rename(columns={col: f'{col} Skill' for col in report.columns})
    report['MAPD'] = np.mean(np.abs(y_prob - baseline_prob))
    return report


def generate_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        odds: pd.DataFrame,
        target_type: TargetType,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> pd.DataFrame:
    """ Generates a report that includes classification metrics, probabilistic metrics and skills (for Result only). """
    metadata_dict = {}

    if start_date is not None:
        metadata_dict['Start Date'] = [start_date]
    if end_date is not None:
        metadata_dict['End Date'] = [end_date]

    metadata_dict['Samples'] = [y_true.shape[0]]
    metadata = pd.DataFrame(data=metadata_dict)

    classification_df = classification_report(y_true=y_true, y_pred=y_pred).round(decimals=3)
    prob_df = probabilistic_report(y_true=y_true, y_prob=y_prob).round(decimals=3)
    profit_balance_df = pd.DataFrame({
        f'MIP-{odd}': [round(metrics.mean_implied_probability(odds=odds[odd]), 3)]
        for odd in ODDS_COLUMNS
    })
    prob_df = pd.concat([prob_df, profit_balance_df], axis=1)

    if target_type == TargetType.RESULT:
        bookmaker_prob = metrics.convert_odds_to_probabilities(odds=odds[['1', 'X', '2']].to_numpy(dtype=np.float64))
        bookmaker_prob_df = probabilistic_report(y_true=y_true, y_prob=bookmaker_prob)
        skill_df = skill_report(
            y_prob=y_prob,
            estimator_prob_report=prob_df,
            baseline_prob=bookmaker_prob,
            baseline_prob_report=bookmaker_prob_df
        ).round(decimals=3)
        report = pd.concat([metadata, classification_df, prob_df, skill_df], axis=1)
    else:
        report = pd.concat([metadata, classification_df, prob_df], axis=1)

    return report


def generate_report_dc(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        odds: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
) -> pd.DataFrame:
    """Generates binary metrics for selected double-chance predictions.

    y_true: actual match result: H, D, A
    y_pred: selected double-chance prediction: 1X, 12, X2
    y_prob: probabilities for [1X, 12, X2]
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), a_min=0.0, a_max=1.0)

    # Map labels to indices.
    result_mapper = {'H': 0, 'D': 1, 'A': 2}
    dc_mapper = {'1X': 0, '12': 1, 'X2': 2}

    result_ids = np.array([result_mapper[label] for label in y_true])
    prediction_ids = np.array([dc_mapper[label] for label in y_pred])

    # Rows: 1X, 12, X2
    # Cols: H, D, A
    membership = np.array([
        [1, 1, 0],  # 1X
        [1, 0, 1],  # 12
        [0, 1, 1]   # X2
    ])
    rows = np.arange(y_true.shape[0])

    # Binary target:
    # 1 -> selected double-chance prediction was correct
    # 0 -> selected double-chance prediction was wrong
    y_true_binary = membership[prediction_ids, result_ids]

    # Since y_pred represents the DC event we selected, prediction is always 1.
    y_pred_binary = np.ones_like(y_true_binary)

    # Probability assigned to the selected DC event.
    selected_prob = y_prob[rows, prediction_ids]
    binary_prob = np.column_stack((1.0 - selected_prob, selected_prob))

    # Bookmaker probabilities for 1X / 12 / X2.
    bookmaker_prob = 1.0 / odds[DOUBLE_CHANCE_ODD_COLUMNS].to_numpy(dtype=np.float64)
    bookmaker_prob = 2.0 * bookmaker_prob / bookmaker_prob.sum(axis=1, keepdims=True)

    selected_bookmaker_prob = bookmaker_prob[rows, prediction_ids]
    binary_bookmaker_prob = np.column_stack((1.0 - selected_bookmaker_prob, selected_bookmaker_prob))

    # Metadata.
    metadata = {'Samples': [y_true.shape[0]]}

    if start_date is not None:
        metadata = {'Start Date': [start_date], **metadata}
    if end_date is not None:
        metadata['End Date'] = [end_date]

    metadata_df = pd.DataFrame(metadata)

    # Classification metrics.
    classification_df = pd.DataFrame({
        'Correct': [int(y_true_binary.sum())],
        'Accuracy': [accuracy_score(y_true_binary, y_pred_binary)],
        'Precision': [precision_score(y_true_binary, y_pred_binary, zero_division=0.0)],
    })

    # Probabilistic metrics.
    probability_df = pd.DataFrame({
        'Log Loss': [log_loss(y_true_binary, binary_prob, labels=[0, 1])],
        'Brier Score': [brier_score_loss(y_true_binary, binary_prob, labels=[0, 1])],
        'RPS': [metrics.ranked_probabilities_score(y_true_binary, binary_prob)]
    })
    profit_balance_df = pd.DataFrame({
        f'MIP-{odd}': [round(metrics.mean_implied_probability(odds=odds[odd]), 3)]
        for odd in DOUBLE_CHANCE_ODD_COLUMNS
    })
    probability_df = pd.concat([probability_df, profit_balance_df], axis=1)

    bookmaker_probability_df = pd.DataFrame({
        'Log Loss': [log_loss(y_true_binary, binary_bookmaker_prob, labels=[0, 1])],
        'Brier Score': [brier_score_loss(y_true_binary, binary_bookmaker_prob, labels=[0, 1])],
        'RPS': [metrics.ranked_probabilities_score(y_true_binary, binary_bookmaker_prob)]
    })

    skill_df = skill_report(
        y_prob=binary_prob,
        estimator_prob_report=probability_df,
        baseline_prob=binary_bookmaker_prob,
        baseline_prob_report=bookmaker_probability_df
    )
    return pd.concat([metadata_df, classification_df, probability_df, skill_df], axis=1).round(3)
