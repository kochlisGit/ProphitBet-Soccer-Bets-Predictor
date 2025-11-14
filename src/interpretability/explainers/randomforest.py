import pandas as pd
from matplotlib.pyplot import Axes
from src.models.model import ClassificationModel
from src.interpretability.explainers.decisiontree import DecisionTreeExplainer


class RandomForestExplainer(DecisionTreeExplainer):
    def __init__(self, model: ClassificationModel, df: pd.DataFrame):
        super().__init__(model=model, df=df)

    def plot_tree_rules(self, max_depth: int = 3, estimator_id: int = 0) -> Axes:
        """ Plot the extracted rules by tree. """

        ax = self._plot_tree(estimator=self._model.get_estimator(estimator_id=estimator_id), max_depth=max_depth)
        return ax
