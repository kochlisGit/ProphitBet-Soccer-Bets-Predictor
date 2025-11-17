import os
import json
import pickle
import shutil
from typing import Any, Dict, Optional, Tuple
from src.models.model import ClassificationModel


class ModelDatabase:
    """ Model database handler class. """

    def __init__(self, league_id: str):
        self._league_id = league_id

        with open('storage/network/models.json', mode='r') as jsonfile:
            models_cfg = json.load(jsonfile)

        # Initialize all available models.
        self._models_directory = models_cfg['models_directory']
        self._models_index_filepath = models_cfg['models_index_filepath']

        # Create model's directory file if it does not exist.
        if not os.path.exists(path=self._models_directory):
            os.makedirs(name=self._models_directory, exist_ok=True)

        # Initialize or Restore index.
        self._index = self._initialize_or_restore_index()

    @property
    def league_id(self) -> str:
        return self._league_id

    @property
    def index(self) -> Dict[str, Dict[str, Any]]:
        return self._index

    def get_model_ids(self) -> list[str]:
        """ Gets a list of all model ids, sorted in ascending order. """

        league_id = self._league_id
        return [] if len(self._index.get(league_id, [])) == 0 else sorted(self._index[league_id].keys())

    def save_model(self, model: ClassificationModel, model_config: Dict[str, Any]):
        """ Saves a model by calling its save method and updates index. """

        # Save model.
        checkpoint_directory = self._build_model_directory(model_id=model.model_id)
        os.makedirs(name=checkpoint_directory, exist_ok=True)
        model.save(checkpoint_directory=checkpoint_directory)

        # Update model index.
        if self._league_id not in self._index:
            self._index[self._league_id] = {}

        self._index[self._league_id][model.model_id] = model_config
        self._save_index()

    def load_model(self, model_id: str) -> Tuple[Optional[ClassificationModel], Optional[Dict[str, Any]]]:
        """ Loads a model by calling its class load method. """

        checkpoint_directory = self._build_model_directory(model_id=model_id)

        if not os.path.exists(path=checkpoint_directory):
            self.delete_model(model_id=model_id)
            return None, None

        # Initialize model instance.
        model_config = self._index[self._league_id][model_id]
        model_cls = model_config['cls']
        model = model_cls(**model_config)

        # Load the built classifier.
        model.load(checkpoint_directory=checkpoint_directory)
        return model, model_config

    def model_exists(self, model_id: str) -> bool:
        if self._league_id not in self._index:
            return False

        return model_id in self._index[self._league_id]

    def load_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        checkpoint_directory = self._build_model_directory(model_id=model_id)

        if not os.path.exists(path=checkpoint_directory):
            self.delete_model(model_id=model_id)
            return None

        return self._index[self._league_id][model_id]

    def update_model_config(self, model_config: Dict[str, Any]):
        """ Updates the model config. """

        self._index[self._league_id][model_config['model_id']] = model_config
        self._save_index()

    def delete_model(self, model_id: str):
        """ Deletes a model from the database and updates the index. """

        # Remove model directory.
        shutil.rmtree(self._build_model_directory(model_id=model_id), ignore_errors=True)

        # Update index.
        league_id = self._league_id

        if model_id in self._index[league_id]:
            self._index[league_id].pop(model_id)
        if len(self._index[league_id]) == 0:
            self._index.pop(league_id)
        self._save_index()

    def delete_league_models(self):
        list(map(self.delete_model, self.get_model_ids()))

    def _initialize_or_restore_index(self) -> Dict[str, Dict[str, Any]]:
        """ Initializes an empty index (Dict) or restores the previously stored index, """

        if not os.path.exists(path=self._models_index_filepath):
            return {}
        else:
            with open(self._models_index_filepath, 'rb') as pklfile:
                index = pickle.load(pklfile)

            validated_index = {}
            for league_id in index:
                validated_index[league_id] = {}

                for model_id, model_config in index[league_id].items():
                    if os.path.exists(path=self._build_model_directory(model_id=model_id, league_id=league_id)):
                        validated_index[league_id][model_id] = model_config

                if len(validated_index[league_id]) == 0:
                    validated_index.pop(league_id)
            return validated_index

    def _save_index(self):
        """ Stores the index to the pre-defined index filepath. The index is stored as a pickle object. """

        with open(self._models_index_filepath, 'wb') as pklfile:
            pickle.dump(self._index, pklfile)

    def _build_model_directory(self, model_id: str, league_id: Optional[str] = None) -> str:
        """ Builds the directory of a model using its league id and model id (user-defined names). """

        if league_id is None:
            league_id = self._league_id

        return f'{self._models_directory}/{league_id}/models/{model_id}'
