import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from database.repositories.model import ModelRepository
from preprocessing.dataset import DatasetPreprocessor
from models.model import ModelConfig, ScikitModel
from models.tasks import ClassificationTask


class Trainer:
    def __init__(self, model_repository: ModelRepository, fit_test_size: int):
        self._model_repository = model_repository
        self._dataset_preprocessor = DatasetPreprocessor()
        self._k_folds = 10
        self._fit_test_size = fit_test_size

    def _train_model(
            self,
            df_train: pd.DataFrame,
            df_test: pd.DataFrame,
            league_id: str,
            model_id: str,
            task: ClassificationTask,
            add_classification_report: bool,
            model_cls: type,
            model_params: dict
    ) -> (ScikitModel, ModelConfig, dict[str, float], str or None):
        model = model_cls(model_id=model_id, **model_params)

        x_train, y_train, normalizer, sampler = self._dataset_preprocessor.preprocess_dataset(
            df=df_train,
            task=task,
            fit_normalizer=True,
            normalizer=model_params.get('normalizer', None),
            sampler=model_params.get('sampler', None),
        )
        x_test, y_test, _, _ = self._dataset_preprocessor.preprocess_dataset(
            df=df_test,
            task=task,
            fit_normalizer=False,
            normalizer=normalizer,
            sampler=None
        )
        evaluation_dict, classification_report = model.fit(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            task=task,
            add_classification_report=add_classification_report,
        )
        model_config = ModelConfig(
            league_id=league_id,
            model_id=model_id,
            model_cls=model_cls,
            task=task,
            model_name=model.model_name
        )
        model_config.calibrate_probabilities = model.calibrate_probabilities
        model_config.normalizer = normalizer
        model_config.sampler = sampler
        return model, model_config, evaluation_dict, classification_report

    def fit(
            self,
            df: pd.DataFrame,
            league_id: str,
            model_id: str,
            task: ClassificationTask,
            save_model: bool,
            model_cls: type,
            model_params: dict
    ) -> (ScikitModel, ModelConfig, dict[str, float], str):
        assert not df.isna().any().any(), 'Cannot preprocess dataframe with nan values'

        df_test = df.iloc[: self._fit_test_size]
        df_train = df.iloc[self._fit_test_size:]

        model, model_config, evaluation_dict, classification_report = self._train_model(
            df_train=df_train,
            df_test=df_test,
            league_id=league_id,
            model_id=model_id,
            task=task,
            add_classification_report=True,
            model_cls=model_cls,
            model_params=model_params
        )

        if save_model:
            self._model_repository.save_model(model=model, model_config=model_config)

        return model, model_config, evaluation_dict, classification_report

    def cross_validate(
            self,
            df: pd.DataFrame,
            league_id: str,
            model_id: str,
            task: ClassificationTask,
            model_cls: type,
            model_params: dict
    ) -> dict[str, float]:
        def get_split_score(input_df: pd.DataFrame, train_ids: np.ndarray, test_ids: np.ndarray) -> dict[str, float]:
            df_train = input_df.iloc[train_ids]
            df_test = input_df.iloc[test_ids]

            model, model_config, evaluation_dict, _ = self._train_model(
                df_train=df_train,
                df_test=df_test,
                league_id=league_id,
                model_id=model_id,
                task=task,
                add_classification_report=False,
                model_cls=model_cls,
                model_params=model_params
            )
            return evaluation_dict

        assert not df.isna().any().any(), 'Cannot preprocess dataframe with nan values'

        x = np.zeros(shape=df.shape[0])
        y = self._dataset_preprocessor.preprocess_targets(df=df, task=task)
        cv_generator = StratifiedKFold(n_splits=self._k_folds, shuffle=True, random_state=0).split(x, y)
        scores = list(map(lambda ids: get_split_score(input_df=df, train_ids=ids[0], test_ids=ids[1]), cv_generator))
        return {
            metric_name: np.mean([evaluation_dict[metric_name] for evaluation_dict in scores])
            for metric_name in scores[0].keys()
        }
