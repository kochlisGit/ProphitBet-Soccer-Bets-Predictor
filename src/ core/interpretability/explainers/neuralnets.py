import numpy as np
import pandas as pd
import shap
from typing import Optional
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainer import ClassifierExplainer


class NeuralNetsExplainer(ClassifierExplainer):
    """ NeuralNets explainer class.

        Supported Classifiers:
            1. NeuralNetwork

        Attributes
        ----------
        :param df: (pd.DataFrame) league's dataframe.
        :param df: (pd.DataFrame) league's dataframe.
        :param model_id: (str) model unique id.
        :param database: (ModelDatabase) active league's model database instance.
    """

    def __init__(self, df: pd.DataFrame, model_id: str, database: ModelDatabase):
        super().__init__(df=df, model_id=model_id, database=database)

        # Validating model.
        if type(self.model).__name__ != 'NeuralNetwork':
            raise NotImplementedError(
                f'NeuralNetsExplainer has not been implemented for classifiers of type: "{type(self.model)}".'
            )

    def _compute_shap_values(self, df: pd.DataFrame) -> shap.Explanation:
        values = df.to_numpy(dtype=np.float32)
        keras_model = self.model.classifier.model
        batch_size = self.model_config.classifier_config['batch_size']
        explainer = shap.GradientExplainer(model=keras_model, data=values, batch_size=batch_size)
        explanation = explainer(values)

        # GradientExplainer does not populate base values, while the shared
        # local-explanation frame requires them for its waterfall plot.
        predictions = np.asarray(keras_model.predict(values, batch_size=batch_size, verbose=0), dtype=np.float64)
        expected_values = np.mean(predictions, axis=0)
        base_values = np.broadcast_to(expected_values, (len(values), *expected_values.shape),)
        return shap.Explanation(
            values=np.asarray(explanation.values),
            base_values=base_values,
            data=values,
            feature_names=self.feature_names
        )

    @property
    def has_attention_weights(self) -> bool:
        """Whether the fitted network exposes a variable-selection model."""

        return getattr(self.model.classifier, 'attn_model', None) is not None

    def get_feature_importances(self, df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Return mean attention weights for each input feature."""

        if not self.has_attention_weights:
            return None

        dataframe = self.train_df if df is None else df
        inputs, _ = self._dataset_preprocessor.construct_inputs(df=dataframe, normalizer=self.normalizer)
        importances = self.model.classifier.get_feature_importances(inputs)

        if importances is None:
            return None

        importances = np.asarray(importances, dtype=np.float64)
        if importances.ndim == 1:
            mean_importances = importances
        else:
            mean_importances = np.mean(importances, axis=tuple(range(importances.ndim - 1)))
        mean_importances = np.asarray(mean_importances).reshape(-1)

        if len(mean_importances) != len(self.feature_names):
            raise ValueError(
                'Attention-weight count does not match the number of model features: '
                f'{len(mean_importances)} and {len(self.feature_names)}.'
            )

        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_importances,
        })
