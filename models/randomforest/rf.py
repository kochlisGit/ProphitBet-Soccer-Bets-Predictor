from models.model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import pickle
import numpy as np


class RandomForest(Model):
    def __init__(
            self,
            input_shape: tuple,
            checkpoint_path: str,
            league_identifier: str,
            model_name: str,
            calibrate_model: bool
    ):
        model_name += '_rf'
        self._calibrate_model = calibrate_model
        super().__init__(
            input_shape=input_shape,
            checkpoint_path=checkpoint_path,
            league_identifier=league_identifier,
            model_name=model_name
        )

    @property
    def calibrate_model(self) -> bool:
        return self._calibrate_model

    def build_model(self, **kwargs):
        n_estimators = kwargs['n_estimators']
        self._model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)

        if self.calibrate_model:
            self._model = CalibratedClassifierCV(RandomForestClassifier(n_jobs=-1), n_jobs=-1)

    def _save(self):
        with open(self.checkpoint_directory, 'wb') as ckp_file:
            pickle.dump(self.model, ckp_file)

    def load(self):
        with open(self.checkpoint_directory, 'rb') as ckp_file:
            self._model = pickle.load(ckp_file)

    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            **kwargs
    ) -> float:
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        return accuracy_score(y_test, y_pred)

    def predict(self, x_inputs: np.ndarray) -> (np.ndarray, np.ndarray):
        predict_proba = np.round(self.model.predict_proba(x_inputs), 2)
        y_pred = self.model.predict(x_inputs)
        return predict_proba, y_pred
