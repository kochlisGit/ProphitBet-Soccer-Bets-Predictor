"""Classifier training dialogs loaded on first access."""

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    'NaiveBayesTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.bayes', 'NaiveBayesTrainerDialog'),
    'RandomForestTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.forest', 'RandomForestTrainerDialog'),
    'LightGBMTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.gbm', 'LightGBMTrainerDialog'),
    'KNearestNeighborsTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.knn', 'KNearestNeighborsTrainerDialog'),
    'DiscriminantAnalysisTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.lda', 'DiscriminantAnalysisTrainerDialog'),
    'LogisticRegressionTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.logistic', 'LogisticRegressionTrainerDialog'),
    'NeuralNetworkTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.nn', 'NeuralNetworkTrainerDialog'),
    'SupportVectorMachineTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.svm', 'SupportVectorMachineTrainerDialog'),
    'DecisionTreeTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.tree', 'DecisionTreeTrainerDialog'),
    'XGBoostTrainerDialog': ('src.gui.components.dialogs.models.train.classifiers.xgb', 'XGBoostTrainerDialog'),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
