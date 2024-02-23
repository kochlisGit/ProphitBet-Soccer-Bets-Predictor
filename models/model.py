import pickle
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from models.tasks import ClassificationTask


class ScikitModel(ABC):
    def __init__(self, model_id: str, model_name: str, calibrate_probabilities: bool, **kwargs):
        self._model_id = model_id
        self._model_name = model_name
        self.calibrate_probabilities = calibrate_probabilities

        self._columns_to_drop = ['Date', 'Season', 'Home Team', 'Away Team', 'HG', 'AG', 'Result']
        self._target_fn = {
            ClassificationTask.Result: lambda df: df['Result'].replace({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32),
            ClassificationTask.Over: lambda df: (df['HG'] + df['AG'] > 2).to_numpy(dtype=np.int64)
        }
        self._model = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model(self) -> BaseEstimator:
        return self._model

    @abstractmethod
    def _build_estimator(self, input_size: int, num_classes: int) -> BaseEstimator:
        pass

    def _build_model(self, input_size: int, num_classes: int):
        estimator = self._build_estimator(input_size=input_size, num_classes=num_classes)

        if self.calibrate_probabilities:
            self._model = CalibratedClassifierCV(estimator, n_jobs=-1)
        else:
            self._model = estimator

    def save(self, checkpoint_directory: str):
        assert self._model is not None, 'Model has not been initialized'

        with open(f'{checkpoint_directory}/model.pkl', mode='wb') as estimator_file:
            pickle.dump(self._model, estimator_file)

    def load(self, checkpoint_directory: str):
        with open(f'{checkpoint_directory}/model.pkl', mode='rb') as estimator_file:
            self._model = pickle.load(estimator_file)

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            task: ClassificationTask,
            add_classification_report: bool
    ) -> (dict[str, float], str):

        if task == ClassificationTask.Result:
            num_classes = 3
        elif task == ClassificationTask.Over:
            num_classes = 2
        else:
            raise NotImplementedError(f'Not supported task: {task.name}')

        self._build_model(input_size=x_train.shape[1], num_classes=num_classes)
        self._model.fit(x_train, y_train)
        return self.evaluate(x=x_test, y=y_test, add_classification_report=add_classification_report)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def evaluate(
            self,
            x: np.ndarray,
            y: np.ndarray,
            add_classification_report: bool
    ) -> (dict[str, float], str or None):
        y_pred = self.predict(x=x)

        metrics = {
            'accuracy': round(accuracy_score(y_true=y, y_pred=y_pred), 2),
            'f1': round(f1_score(y_true=y, y_pred=y_pred, average='macro', zero_division=0.0), 2),
            'precision': round(precision_score(y_true=y, y_pred=y_pred, average='macro', zero_division=0.0), 2),
            'recall': round(recall_score(y_true=y, y_pred=y_pred, average='macro', zero_division=0.0), 2)
        }

        if add_classification_report:
            report = classification_report(y_true=y, y_pred=y_pred)
        else:
            report = None

        return metrics, report


class ModelConfig:
    def __init__(
            self,
            league_id: str,
            model_id: str,
            model_cls: type,
            task: ClassificationTask,
            model_name: str
    ):
        self._league_id = league_id
        self._model_id = model_id
        self._model_cls = model_cls
        self._task = task
        self._model_name = model_name

        self.calibrate_probabilities = False
        self.sampler = None
        self.normalizer = None
        self.home_fixture_percentile = (0, 0.0)
        self.draw_fixture_percentile = (0, 0.0)
        self.away_fixture_percentile = (0, 0.0)
        self.under_fixture_percentile = (0, 0.0)
        self.over_fixture_percentile = (0, 0.0)

    @property
    def league_id(self) -> str:
        return self._league_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_cls(self) -> type:
        return self._model_cls

    @property
    def task(self) -> ClassificationTask:
        return self._task

    @property
    def model_name(self) -> str:
        return self._model_name
