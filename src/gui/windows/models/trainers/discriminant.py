import pandas as pd
from typing import Dict, Any, Type
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.windows.models.trainer import TrainerDialog
from src.models.classifiers.discriminant import DiscriminantAnalysisClassifier


class DiscriminantTrainerDialog(TrainerDialog):
    """ LDA/QDA trainer window. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._neighbors_step = 6
        self._oas_options = {'Yes': True, 'No': False}
        self._boundaries = {'Linear': 'linear', 'Quadratic': 'quadratic'}

        self._combo_oas = None
        self._combo_boundaries = None

        super().__init__(
            df=df,
            model_db=model_db,
            title='Discriminant Analysis Regression Trainer',
            width=800,
            height=250,
            supports_calibration=False
        )

    def get_model_cls(self) -> Type:
        return DiscriminantAnalysisClassifier

    def _add_trainer_widgets(self, root: QVBoxLayout):
        row1_box = QHBoxLayout()
        row1_box.setContentsMargins(0, 10, 0, 0)
        row1_box.addStretch(1)

        self._combo_oas = QComboBox()
        self._combo_oas.setFixedWidth(90)
        for oas in self._oas_options:
            self._combo_oas.addItem(oas)
        self._add_tunable_param(
            name=' Oracle Approximating Shrinkage (OAS)',
            placeholder_name='oas',
            widget=self._combo_oas,
            layout=row1_box,
            tooltip='Whether to apply OAS optimization.'
        )

        self._combo_boundaries = QComboBox()
        self._combo_boundaries.setFixedWidth(120)
        for boundary in self._boundaries:
            self._combo_boundaries.addItem(boundary)
        self._add_tunable_param(
            name=' Decision Boundaries',
            placeholder_name='decision_boundary',
            widget=self._combo_boundaries,
            layout=row1_box,
            tooltip='Decision-Boundary algorithm.'
        )

        row1_box.addStretch(1)
        root.addLayout(row1_box)

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            'oas': self._oas_options[self._combo_oas.currentText()],
            'decision_boundary': self._boundaries[self._combo_boundaries.currentText()]
        }
