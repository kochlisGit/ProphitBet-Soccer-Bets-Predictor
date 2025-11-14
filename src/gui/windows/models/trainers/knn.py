import pandas as pd
from typing import Dict, Any, Type
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QSlider, QVBoxLayout
from superqt import QLabeledSlider
from src.database.model import ModelDatabase
from src.gui.widgets.sliders import add_snap_behavior
from src.gui.windows.models.trainer import TrainerDialog
from src.models.classifiers.knn import KNN


class KNNTrainerDialog(TrainerDialog):
    """ KNN trainer window. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._neighbors_step = 6
        self._weights = {'Uniform': 'uniform', 'Distance': 'distance'}
        self._distances = {'Manhattan': 1, 'Euclidean': 2}

        self._slider_neighbors = None
        self._combo_weights = None
        self._combo_distances = None

        super().__init__(
            df=df,
            model_db=model_db,
            title='KNN Trainer',
            width=800,
            height=250,
            supports_calibration=True
        )

    def get_model_cls(self) -> Type:
        return KNN

    def _add_trainer_widgets(self, root: QVBoxLayout):
        row1_box = QHBoxLayout()
        row1_box.setContentsMargins(0, 10, 0, 0)
        row1_box.addStretch(1)

        self._slider_neighbors = QLabeledSlider(Qt.Orientation.Horizontal)
        self._slider_neighbors.setFixedWidth(150)
        self._slider_neighbors.setRange(3, 99)
        self._slider_neighbors.setSingleStep(6)
        self._slider_neighbors.setTickInterval(12)
        self._slider_neighbors.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_neighbors.setValue(15)
        add_snap_behavior(slider=self._slider_neighbors, step=6)
        self._add_tunable_param(
            name='Neighbors',
            placeholder_name='n_neighbors',
            widget=self._slider_neighbors,
            layout=row1_box,
            tooltip='Number of closest instances (neighbors).'
        )

        self._combo_weights = QComboBox()
        self._combo_weights.setFixedWidth(100)
        for weight in self._weights:
            self._combo_weights.addItem(weight)
        self._combo_weights.setCurrentIndex(1)
        self._add_tunable_param(
            name='Neighbor Weights',
            placeholder_name='weights',
            widget=self._combo_weights,
            layout=row1_box,
            tooltip='The weights of KNN neighbors. If uniform, then all distanced are equally weighted.'
        )

        self._combo_distances = QComboBox()
        self._combo_distances.setFixedWidth(100)
        for metric in self._distances:
            self._combo_distances.addItem(metric)
        self._combo_distances.setCurrentIndex(1)
        self._add_tunable_param(
            name='Distance Metric',
            placeholder_name='p',
            widget=self._combo_distances,
            layout=row1_box,
            tooltip='Distance metric between 2 instances.'
        )

        row1_box.addStretch(1)
        root.addLayout(row1_box)

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            'n_neighbors': self._slider_neighbors.value(),
            'weights': self._weights[self._combo_weights.currentText()],
            'p': self._distances[self._combo_distances.currentText()]
        }
