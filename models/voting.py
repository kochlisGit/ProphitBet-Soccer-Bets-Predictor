import pickle
import numpy as np
from scipy.stats import mode
from models.tasks import ClassificationTask
from models.model import ScikitModel, ModelConfig
from database.repositories.model import ModelRepository
from preprocessing.dataset import DatasetPreprocessor


class VotingModel(ScikitModel):
    def __init__(
            self,
            model_id: str,
            model_configs: list[ModelConfig] or None = None,
            calibrate_probabilities: bool = False,
            **kwargs
    ):
        assert not calibrate_probabilities, 'Probability calibration is not supported in this model'

        super().__init__(
            model_id=model_id,
            model_name='voting',
            calibrate_probabilities=False,
            **kwargs
        )

        self._model_configs = model_configs
        self._model_repository = kwargs.get('model_repository')

        if model_configs is not None:
            self._models = [
                self._model_repository.load_model(model_config=model_config)
                for model_config in self._model_configs
            ]
        else:
            self._models = None

        self._dataset_preprocessor = DatasetPreprocessor()

    def _build_estimator(self, input_size: int, num_classes: int) -> None:
        raise NotImplementedError('Train function is not supported for Voting class')

    def save(self, checkpoint_directory: str):
        assert self._model_configs is not None, 'Voting model has not been created. Model Config list is None.'

        with open(f'{checkpoint_directory}/model.pkl', mode='wb') as estimator_file:
            pickle.dump(self._model_configs, estimator_file)

    def load(self, checkpoint_directory: str):
        with open(f'{checkpoint_directory}/model.pkl', mode='rb') as estimator_file:
            self._model_configs = pickle.load(estimator_file)

        self._models = [
            self._model_repository.load_model(model_config=model_config)
            for model_config in self._model_configs
        ]

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            task: ClassificationTask,
            add_classification_report: bool
    ) -> (dict[str, float], str or None):
        return self.evaluate(
            x=x_test,
            y=y_test,
            add_classification_report=add_classification_report
        )

    def _preprocess_inputs(self, x: np.ndarray, config: ModelConfig) -> np.ndarray:
        x, normalizer = self._dataset_preprocessor.normalize_inputs(x=x, normalizer=config.normalizer, fit=False)
        return x

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        y_pred_proba_list = [
            model.predict_proba(x=self._preprocess_inputs(x=x, config=config))
            for model, config in zip(self._models, self._model_configs)
        ]
        return np.mean(y_pred_proba_list, axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred_classes = [
            model.predict(x=self._preprocess_inputs(x=x, config=config)).reshape((-1, 1))
            for model, config in zip(self._models, self._model_configs)
        ]
        return mode(pred_classes, axis=0).mode.flatten()
