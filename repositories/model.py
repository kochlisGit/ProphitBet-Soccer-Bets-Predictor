import os
import pickle
import shutil
from models.model import ScikitModel, ModelConfig


class ModelRepository:
    def __init__(self, models_directory: str, models_index_filepath: str):
        self._models_directory = models_directory
        self._models_index_filepath = models_index_filepath

        os.makedirs(name=models_directory, exist_ok=True)

        if not os.path.exists(models_index_filepath):
            self._index = {}
        else:
            self._load_index()

    @property
    def index(self) -> dict[str, dict[str, dict[str, ModelConfig]]]:
        return self._index

    def _save_index(self):
        with open(self._models_index_filepath, 'wb') as pklfile:
            pickle.dump(self._index, pklfile)

    def _load_index(self):
        with open(self._models_index_filepath, 'rb') as pklfile:
            self._index = pickle.load(pklfile)

    def get_model_configs(self, league_id: str) -> dict[str, dict[str, ModelConfig]]:
        return {} if league_id not in self._index else self._index[league_id]

    def update_model_config(self, model_config: ModelConfig):
        self._index[model_config.league_id][model_config.task.name][model_config.model_id] = model_config
        self._save_index()

    def save_model(self, model: ScikitModel, model_config: ModelConfig):
        if model_config.league_id not in self._index:
            self._index.update({model_config.league_id: {}})

        if model_config.task.name not in self._index[model_config.league_id]:
            self._index[model_config.league_id].update({model_config.task.name: {}})

        self._index[model_config.league_id][model_config.task.name][model_config.model_id] = model_config
        self._save_index()

        checkpoint_directory = f'{self._models_directory}/{model_config.league_id}/{model_config.task.name}/{model.model_id}'
        os.makedirs(name=checkpoint_directory, exist_ok=True)
        model.save(checkpoint_directory=checkpoint_directory)

    def load_model(self, model_config: ModelConfig) -> ScikitModel:
        checkpoint_directory = f'{self._models_directory}/{model_config.league_id}/{model_config.task.name}/{model_config.model_id}'
        model = model_config.model_cls(
            model_id=model_config.model_id,
            calibrate_probabilities=model_config.calibrate_probabilities,
            model_repository=self
        )
        model.load(checkpoint_directory=checkpoint_directory)
        return model

    def delete_model(self, model_config: ModelConfig):
        del self._index[model_config.league_id][model_config.task.name][model_config.model_id]
        shutil.rmtree(
            path=f'{self._models_directory}/{model_config.league_id}/{model_config.task.name}/{model_config.model_id}',
            ignore_errors=True
        )

        if len(self._index[model_config.league_id][model_config.task.name]) == 0:
            del self._index[model_config.league_id][model_config.task.name]
            shutil.rmtree(
                path=f'{self._models_directory}/{model_config.league_id}/{model_config.task.name}',
                ignore_errors=True
            )

        if len(self._index[model_config.league_id]) == 0:
            del self._index[model_config.league_id]
            shutil.rmtree(path=f'{self._models_directory}/{model_config.league_id}', ignore_errors=True)

        self._save_index()

    def delete_league_models(self, league_id: str):
        if league_id in self._index:
            shutil.rmtree(path=f'{self._models_directory}/{league_id}', ignore_errors=True)
            del self._index[league_id]
            self._save_index()
