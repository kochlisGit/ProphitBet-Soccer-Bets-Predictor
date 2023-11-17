import os
import shutil
from models.model import Model
from models.tf.nn import FCNet
from models.scikit.rf import RandomForest


class ModelRepository:
    def __init__(self, models_checkpoint_directory: str):
        self._models_checkpoint_directory = models_checkpoint_directory

        os.makedirs(name=models_checkpoint_directory, exist_ok=True)

    def store_model(self, model: Model, league_name: str):
        checkpoint_directory = f"{self._models_checkpoint_directory}/{league_name}"
        os.makedirs(name=checkpoint_directory, exist_ok=True)

        model.save(f"{checkpoint_directory}/{model.get_model_name()}")

    def get_all_models(self, league_name: str) -> list or None:
        checkpoint_directory = f"{self._models_checkpoint_directory}/{league_name}/"
        if not os.path.exists(checkpoint_directory):
            return None
        else:
            model_names = os.listdir(checkpoint_directory)
            return [name.split(".")[0] for name in model_names]

    def load_model(
        self,
        league_name: str,
        model_name: str,
        input_shape: tuple,
        random_seed: int,
    ) -> Model or None:
        if model_name == "rf":
            model_name += ".pickle"
        checkpoint_filepath = (
            f"{self._models_checkpoint_directory}/{league_name}/{model_name}"
        )

        if os.path.exists(checkpoint_filepath):
            if model_name == "nn":
                model = FCNet(input_shape=input_shape, random_seed=random_seed)
            elif model_name == "rf.pickle":
                model = RandomForest(input_shape=input_shape, random_seed=random_seed)
            else:
                raise NotImplementedError(
                    f'Type of model "{model_name}" has not been implemented yet'
                )
        else:
            return None

        model.load(checkpoint_filepath=checkpoint_filepath)
        return model

    def delete_model(self, league_name: str, model_name: str) -> bool:
        checkpoint_filepath = (
            f"{self._models_checkpoint_directory}/{league_name}/{model_name}"
        )

        if os.path.exists(checkpoint_filepath):
            shutil.rmtree(checkpoint_filepath) if os.path.isdir(
                checkpoint_filepath
            ) else os.remove(checkpoint_filepath)
            return True
        else:
            return False
