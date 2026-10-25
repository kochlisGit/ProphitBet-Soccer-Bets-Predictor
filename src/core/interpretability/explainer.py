import os
import pickle
import numpy as np
import pandas as pd
import shap
from abc import ABC, abstractmethod
from typing import Optional
from src.core.database.model import ModelDatabase
from src.core.preprocessing.dataset import DatasetPreprocessor, construct_targets


class ClassifierExplainer(ABC):
    """ Basic explainer class for classification models.

        Attributes
        ----------
        :param df: (pd.DataFrame) league's dataframe.
        :param model_id: (str): model unique id.
        :param database: (ModelDatabase) active league's model database instance.
    """

    def __init__(self, df: pd.DataFrame, model_id: str, database: ModelDatabase):
        self.model, self.model_config = database.load_model(model_id=model_id)

        if self.model is None:
            raise ValueError('Pre-trained classifier is None. Call model.fit first, before initializing the explainer.')

        self.target_type = self.model_config.classifier_config['target_type']
        self.normalizer = self.model_config.classifier_config['normalizer']

        # Preprocessing dataframe.
        self.valid_df = df.dropna().reset_index(drop=True)

        self._dataset_preprocessor = DatasetPreprocessor()
        self._metadata_columns = self._dataset_preprocessor.metadata_columns
        self.target_names = list(self.target_type.get_mapper().keys())
        self.feature_names = self.valid_df.columns.drop(self._metadata_columns).tolist()

        # Computing shap values.
        self.train_df = self.valid_df.iloc[-self.model_config.num_train_samples:]
        self.test_df = self.valid_df.iloc[: df.shape[0] - self.model_config.num_train_samples]

        # Fetching explanation directory.
        model_path = database.build_model_directory(
            model_id=self.model_config.model_id,
            league_id=self.model_config.league_id
        )
        self._explanation_fp = f'{model_path}/shap.pkl'

        # Declaring shap values and cluster placeholders.
        self._train_shap_values = None
        self._train_cluster = None
        self._eval_shap_values = None
        self._eval_cluster = None

    @property
    def train_shap_values(self) -> Optional[shap.Explanation]:
        return self._train_shap_values

    @property
    def train_cluster(self) -> Optional[np.ndarray]:
        return self._train_cluster

    @property
    def eval_shap_values(self) -> Optional[shap.Explanation]:
        return self._eval_shap_values

    @property
    def eval_cluster(self) -> Optional[np.ndarray]:
        return self._eval_cluster

    @abstractmethod
    def _compute_shap_values(self, df: pd.DataFrame) -> shap.Explanation:
        pass

    def compute_shap_values(self):
        """ Computes SHAP values and feature cluster for train and test and stores them inside model's directory. """

        if os.path.exists(self._explanation_fp):
            with open(self._explanation_fp, mode='rb') as pickle_file:
                explanation_dict = pickle.load(pickle_file)
        else:
            x_train_values, _ = self._dataset_preprocessor.construct_inputs(df=self.train_df, normalizer=self.normalizer,)
            x_train = pd.DataFrame(data=x_train_values, columns=self.feature_names)
            y_train = construct_targets(df=self.train_df, target_type=self.target_type, one_hot=False)
            x_test_values, _ = self._dataset_preprocessor.construct_inputs(df=self.test_df, normalizer=self.normalizer,)
            x_test = pd.DataFrame(data=x_test_values, columns=self.feature_names)
            y_test = construct_targets(df=self.test_df, target_type=self.target_type, one_hot=False)

            explanation_dict = {
                'train': {
                    'shap_values': self._compute_shap_values(df=x_train),
                    'cluster': shap.utils.hclust(x_train, y_train)
                },
                'test': {
                    'shap_values': self._compute_shap_values(df=x_test),
                    'cluster': shap.utils.hclust(x_test, y_test)
                }
            }
            with open(self._explanation_fp, mode='wb') as pickle_file:
                pickle.dump(explanation_dict, pickle_file)

        train_explanation = explanation_dict['train']
        self._train_shap_values = train_explanation['shap_values']
        self._train_cluster = train_explanation['cluster']

        test_explanation = explanation_dict['test']
        self._eval_shap_values = test_explanation['shap_values']
        self._eval_cluster = test_explanation['cluster']
