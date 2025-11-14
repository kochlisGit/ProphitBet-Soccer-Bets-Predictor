import pandas as pd
from typing import Type
from PyQt6.QtWidgets import QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.windows.models.explainer import ExplainerDialog
from src.interpretability.explainers.naivebayes import NaiveBayesExplainer
from src.models.classifiers.naivebayes import NaiveBayes


class NaiveBayesExplainerDialog(ExplainerDialog):
    """ Class that supports Naive Bayes explanations. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._visualize_btn = None

        super().__init__(df=df, model_db=model_db, title='Naive Bayes Explainer', width=400, height=600)

    def _get_model_cls(self) -> Type:
        return NaiveBayes

    def _add_additional_widgets(self, root: QVBoxLayout):
        return

    def _get_explainer(self, model: NaiveBayes) -> NaiveBayesExplainer:
        return NaiveBayesExplainer(model=model, df=self._df)
