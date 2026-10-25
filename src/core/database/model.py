import os
import json
import pickle
import shutil
import optuna
from typing import Any, Dict, Optional, Tuple
from src.core.models.dashboard import get_study_filepath, get_storage_url
from src.core.models.model import Classifier, ModelConfig


_NEURAL_NETWORK_CLASS_PATH = 'src.core.models.classifiers.neuralnets.nn:NeuralNetwork'


class _ModelIndexUnpickler(pickle.Unpickler):
    """Load model metadata without importing TensorFlow-backed classifiers."""

    def find_class(self, module: str, name: str) -> Any:
        if module == 'src.core.models.classifiers.neuralnets.nn' and name == 'NeuralNetwork':
            return _NEURAL_NETWORK_CLASS_PATH
        return super().find_class(module, name)


class ModelDatabase:
    """ Model database handler class.

        Functions:
            1. Loads all stored models.
            2. Checks whether a model exists by id.
            3. Stores a trained model and its config.
            4. Loads a trained model and its config.
            5. Update model configs.
            7. Deletes specified model by id.

        Attributes
        ----------
        :param league_id: (str) current league id.
    """

    def __init__(self, league_id: str):
        self._league_id = league_id

        with open('storage/configs/models.json', mode='r', encoding='utf-8') as jsonfile:
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

    def _initialize_or_restore_index(self) -> Dict[str, Dict[str, Any]]:
        """ Initializes an empty index (Dict) or restores the previously stored index, """

        if not os.path.exists(path=self._models_index_filepath):
            return {}
        else:
            with open(self._models_index_filepath, 'rb') as pklfile:
                index = _ModelIndexUnpickler(pklfile).load()

            validated_index = {}
            for league_id in index:
                stored_models = {
                    model_id: model_config
                    for model_id, model_config in index[league_id].items()
                    if os.path.exists(path=self.build_model_directory(model_id=model_id, league_id=league_id))
                }
                if stored_models:
                    validated_index[league_id] = stored_models

            return validated_index

    def _update_index(self):
        """ Updates the model index and stores it in the pre-defined path. """

        with open(self._models_index_filepath, 'wb') as pklfile:
            pickle.dump(self._index, pklfile)

    def build_model_directory(self, model_id: str, league_id: Optional[str] = None) -> str:
        """ Builds the directory of a model using its league id and model id (user-defined names). """

        if league_id is None:
            league_id = self._league_id

        return f'{self._models_directory}/{league_id}/models/{model_id}'

    def get_model_ids(self) -> list[str]:
        """ Gets a list of all model ids, sorted in ascending order. """

        model_ids = self._index.get(self._league_id, [])
        return [] if len(model_ids) == 0 else list(model_ids.keys())

    def model_exists(self, model_id: str) -> bool:
        """ Checks whether a specified model exists by id. """

        return self._league_id in self._index and model_id in self._index[self._league_id]

    def save_model(self, model_config: ModelConfig, classifier: Classifier):
        """ Saves a model by calling its save method and updates index.

            Args
            ----
            :param model_config: (ModelConfig) model metadata and classifier configuration.
            :param classifier: (Classifier) trained classifier to save.
        """

        # Saving model.
        checkpoint_directory = self.build_model_directory(model_id=model_config.model_id)
        os.makedirs(name=checkpoint_directory, exist_ok=True)
        classifier.save(checkpoint_directory=checkpoint_directory)

        # Updating index.
        if self._league_id not in self._index:
            self._index[self._league_id] = {}

        self._index[self._league_id][model_config.model_id] = model_config
        self._update_index()

    def load_model(self, model_id: str) -> Tuple[Classifier, ModelConfig]:
        """ Reconstructs and loads a classifier, returning it with its stored config. """

        # Loading model config.
        model_config = self._index[self._league_id][model_id]

        # Re-initializing the wrapper and its underlying classifier.
        classifier_config = model_config.classifier_config
        if classifier_config.get('model_cls') == _NEURAL_NETWORK_CLASS_PATH:
            from src.core.models.classifiers.neuralnets.nn import NeuralNetwork

            classifier_config = {**classifier_config, 'model_cls': NeuralNetwork}
        classifier = Classifier.from_config(model_config=classifier_config)
        classifier._classifier = classifier.build(
            input_size=classifier_config['input_size'],
            num_classes=classifier_config['num_classes']
        )

        # Loading trained classifier.
        checkpoint_directory = self.build_model_directory(model_id=model_id)
        classifier.load(checkpoint_directory=checkpoint_directory)
        return classifier, model_config

    def update_model_config(self, model_config: ModelConfig):
        """ Updates an existing model config. """

        self._index[self._league_id][model_config.model_id] = model_config
        self._update_index()

    def delete_model(self, model_id: str):
        """ Deletes an existing model and its study using its id. """

        shutil.rmtree(self.build_model_directory(model_id=model_id), ignore_errors=True)
        self._index[self._league_id].pop(model_id)
        self._update_index()

        study_filepath = get_study_filepath(league_id=self._league_id, entity='models')

        if os.path.exists(study_filepath):
            storage_url = get_storage_url(study_filepath=study_filepath)

            for study_name in optuna.get_all_study_names(storage=storage_url):
                if study_name == model_id:
                    try:
                        optuna.delete_study(study_name=model_id, storage=storage_url)
                    except KeyError as _:
                        return
                    finally:
                        break
