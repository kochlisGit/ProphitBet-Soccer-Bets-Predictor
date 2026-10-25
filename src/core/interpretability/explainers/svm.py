import numpy as np
import pandas as pd
import shap
from typing import Optional
from shap.maskers import Independent
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainer import ClassifierExplainer


class SupportVectorMachineExplainer(ClassifierExplainer):
    """ Explainer class for SVM models.

        Attributes
        ----------
        :param df: (pd.DataFrame) league's dataframe.
        :param model_id: (str): model unique id.
        :param database: (ModelDatabase) active league's model database instance.
    """

    def __init__(self, df: pd.DataFrame, model_id: str, database: ModelDatabase):
        super().__init__(df=df, model_id=model_id, database=database)
        self._kernel = self.model_config.classifier_config['kernel']

    def _compute_shap_values(self, df: pd.DataFrame) -> shap.Explanation:
        clf = self.model.classifier

        if self._kernel is None or self._kernel == 'linear':
            if self.model_config.classifier_config['calibrate_probabilities']:
                clf = clf.calibrated_classifiers_[0].estimator
            explainer = shap.LinearExplainer(model=clf, masker=Independent(df, max_samples=df.shape[0]))
            return explainer(df)

        def predict_probabilities(values) -> np.ndarray:
            # PermutationExplainer requires a callable model. Converting to an
            # ndarray also avoids sklearn feature-name warnings because the
            # stored estimator was fitted on preprocessed arrays.
            return clf.predict_proba(np.asarray(values))

        masker = Independent(df, max_samples=min(df.shape[0], 100))
        explainer = shap.PermutationExplainer(model=predict_probabilities, masker=masker, feature_names=df.columns.tolist())
        return explainer(df, max_evals=2 * df.shape[1] + 1)

    def get_coefficients(self) -> Optional[pd.DataFrame]:
        if self._kernel is not None and self._kernel != 'linear':
            return None

        coefficients = self.model.get_coefficients()

        if len(self.target_names) < 3:
            return pd.DataFrame({'Feature': self.feature_names, 'Coefficient': coefficients})
        else:
            return pd.DataFrame(data=coefficients, columns=self.feature_names, index=self.target_names)
