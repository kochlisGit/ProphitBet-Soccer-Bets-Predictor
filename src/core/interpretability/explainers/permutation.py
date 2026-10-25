import numpy as np
import pandas as pd
import shap
from shap.maskers import Independent
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainer import ClassifierExplainer


class PermutationExplainer(ClassifierExplainer):
    """ Permutation explainer class for KNN, Naive Bayes and non-linear/non-tree models.

        Attributes
        ----------
        :param df: (pd.DataFrame) league's dataframe.
        :param model_id: (str): model unique id.
        :param database: (ModelDatabase) active league's model database instance.
    """

    def __init__(self, df: pd.DataFrame, model_id: str, database: ModelDatabase):
        super().__init__(df=df, model_id=model_id, database=database)

    def _compute_shap_values(self, df: pd.DataFrame) -> shap.Explanation:
        clf = self.model.classifier

        def predict_probabilities(values) -> np.ndarray:
            return clf.predict_proba(np.asarray(values))

        masker = Independent(df, max_samples=min(df.shape[0], 100))
        explainer = shap.PermutationExplainer(model=predict_probabilities, masker=masker, feature_names=df.columns.tolist())
        return explainer(df, max_evals=2 * df.shape[1] + 1)
