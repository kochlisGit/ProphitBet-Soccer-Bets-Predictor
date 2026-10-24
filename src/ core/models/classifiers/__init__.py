"""Classifier exports loaded only when the classifier is requested."""

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    'NaiveBayes': ('src.core.models.classifiers.bayes', 'NaiveBayes'),
    'RandomForest': ('src.core.models.classifiers.forest', 'RandomForest'),
    'KNearestNeighbors': ('src.core.models.classifiers.knn', 'KNearestNeighbors'),
    'DiscriminantAnalysis': ('src.core.models.classifiers.lda', 'DiscriminantAnalysis'),
    'LogisticRegressor': ('src.core.models.classifiers.logistic', 'LogisticRegressor'),
    'SupportVectorMachine': ('src.core.models.classifiers.svm', 'SupportVectorMachine'),
    'DecisionTree': ('src.core.models.classifiers.tree', 'DecisionTree'),
    'XGBoost': ('src.core.models.classifiers.xgb', 'XGBoost'),
    'LightGBM': ('src.core.models.classifiers.gbm', 'LightGBM'),
}

__all__ = [*_EXPORTS, 'supported_classifiers']


def _load_classifier(name: str):
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __getattr__(name: str):
    if name == 'supported_classifiers':
        classifiers = [
            _load_classifier(classifier_name)
            for classifier_name in _EXPORTS
            if classifier_name != 'LightGBM'
        ]
        try:
            classifiers.append(_load_classifier('LightGBM'))
        except ImportError:
            pass
        globals()[name] = classifiers
        return classifiers

    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    if name == 'LightGBM':
        try:
            return _load_classifier(name)
        except ImportError:
            globals()[name] = None
            return None

    return _load_classifier(name)
