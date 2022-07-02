from abc import ABC, abstractmethod
import numpy as np
import os


class Model(ABC):
    def __init__(self, input_shape: tuple, checkpoint_path: str, league_identifier: str, model_name: str):
        self._input_shape = input_shape
        self._checkpoint_path = checkpoint_path
        self._league_identifier = league_identifier
        self._model_name = model_name
        self._model = None

    @property
    def input_shape(self) -> tuple:
        return self._input_shape

    @property
    def checkpoint_directory(self):
        return f'{self._checkpoint_path}/{self._league_identifier}/{self._model_name}'

    @property
    def model(self):
        return self._model

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    def save(self):
        save_dir = f'{self._checkpoint_path}/{self._league_identifier}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self._save()

    @abstractmethod
    def _save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            **kwargs
    ) -> float:
        pass

    @abstractmethod
    def predict(self, x_inputs: np.ndarray) -> (np.ndarray, np.ndarray):
        pass
