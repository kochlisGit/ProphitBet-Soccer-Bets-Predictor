import pandas as pd
from typing import Optional
from PySide6.QtWidgets import QWidget
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainers.permutation import PermutationExplainer
from src.gui.components.dialogs.models.interpretability.explainers.base import ExplainerBaseDialog


class PermutationExplainerDialog(ExplainerBaseDialog):
    """SHAP-only dialog for classifiers explained by feature permutation."""

    def __init__(
            self,
            df: pd.DataFrame,
            model_id: str,
            model_database: ModelDatabase,
            parent: Optional[QWidget] = None,
    ):
        explainer = PermutationExplainer(df=df, model_id=model_id, database=model_database)
        super().__init__(explainer=explainer, parent=parent)
        self.setWindowTitle(model_id)
