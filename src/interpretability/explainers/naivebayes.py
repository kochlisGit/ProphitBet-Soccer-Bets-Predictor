import pandas as pd
import shap
from typing import Optional
from src.models.model import ClassificationModel
from src.interpretability.explainer import ClassifierExplainer


class NaiveBayesExplainer(ClassifierExplainer):
    def __init__(self, model: ClassificationModel, df: pd.DataFrame):
        super().__init__(model=model, df=df)

    def _compute_shap_values(self) -> Optional[shap.Explanation]:
        """ Returns shap values of kernel explainer. """

        return None
