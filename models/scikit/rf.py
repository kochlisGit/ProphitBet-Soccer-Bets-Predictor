import pickle
import numpy as np
from models.model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


class RandomForest(Model):
    def __init__(self, input_shape: tuple, random_seed: int):
        super().__init__(input_shape=input_shape, random_seed=random_seed)

    def get_model_name(self) -> str:
        return "rf.pickle"

    def _build_model(self, **kwargs):
        n_estimators = kwargs["n_estimators"]
        max_features = kwargs["max_features"]
        max_depth = kwargs["max_depth"]
        min_samples_leaf = kwargs["min_samples_leaf"]
        min_samples_split = kwargs["min_samples_split"]
        bootstrap = kwargs["bootstrap"]
        class_weight = kwargs["class_weight"]
        is_calibrated = kwargs["is_calibrated"]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=self.random_seed,
        )
        return (
            CalibratedClassifierCV(base_estimator=rf, cv=5, n_jobs=-1)
            if is_calibrated
            else rf
        )

    def save(self, checkpoint_filepath: str):
        with open(checkpoint_filepath, "wb") as ckp_file:
            pickle.dump(self.model, ckp_file)

    def _load(self, checkpoint_filepath: str):
        with open(checkpoint_filepath, "rb") as ckp_file:
            return pickle.load(ckp_file)

    def _train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        self.model.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        predict_prob = np.round(self.model.predict_proba(x), 2)
        y_pred = self.model.predict(x)
        return y_pred, predict_prob
