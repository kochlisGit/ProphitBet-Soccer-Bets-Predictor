import pandas as pd
from typing import Dict, Any, Type
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.windows.models.trainer import TrainerDialog
from src.models.classifiers.naivebayes import NaiveBayes


class NaiveBayesTrainerDialog(TrainerDialog):
    """ Naive Bayes trainer window. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._algorithms = {'Gaussian': 'gaussian', 'Multinomial': 'multinomial', 'Complement': 'complement'}

        self._combo_algorithm = None

        super().__init__(
            df=df,
            model_db=model_db,
            title='Naive Bayes Trainer',
            width=800,
            height=250,
            supports_calibration=True
        )

    def get_model_cls(self) -> Type:
        return NaiveBayes

    def _add_trainer_widgets(self, root: QVBoxLayout):
        row1_box = QHBoxLayout()
        row1_box.setContentsMargins(0, 10, 0, 0)
        row1_box.addStretch(1)

        self._combo_algorithm = QComboBox()
        self._combo_algorithm.setFixedWidth(150)
        for algorithm in self._algorithms:
            self._combo_algorithm.addItem(algorithm)
        self._add_tunable_param(
            name='Algorithm',
            placeholder_name='algorithm',
            widget=self._combo_algorithm,
            layout=row1_box,
            tooltip='Naive Bayes algorithm for continuous variables.'
        )

        row1_box.addStretch(1)
        root.addLayout(row1_box)

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            'algorithm': self._algorithms[self._combo_algorithm.currentText()]
        }
