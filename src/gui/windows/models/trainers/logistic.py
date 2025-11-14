import pandas as pd
from typing import Dict, Any, Type
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.windows.models.trainer import TrainerDialog
from src.models.classifiers.logistic import LogisticRegressor


class LogisticRegressionTrainerDialog(TrainerDialog):
    """ Logistic regression trainer window. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._penalties = [None, 'l1', 'l2']

        self._combo_penalty = None

        super().__init__(
            df=df,
            model_db=model_db,
            title='Logistic Regression Trainer',
            width=800,
            height=250,
            supports_calibration=True
        )

    def get_model_cls(self) -> Type:
        return LogisticRegressor

    def _add_trainer_widgets(self, root: QVBoxLayout):
        row1_box = QHBoxLayout()
        row1_box.setContentsMargins(0, 10, 0, 0)
        row1_box.addStretch(1)

        self._combo_penalty = QComboBox()
        self._combo_penalty.setFixedWidth(80)
        for penalty in self._penalties:
            self._combo_penalty.addItem(str(penalty))
        self._add_tunable_param(
            name='Penalty',
            placeholder_name='penalty',
            widget=self._combo_penalty,
            layout=row1_box,
            tooltip='Whether to apply penalty: None, l1 or l2.'
        )

        row1_box.addStretch(1)
        root.addLayout(row1_box)

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            'penalty': self._penalties[self._combo_penalty.currentIndex()]
        }
