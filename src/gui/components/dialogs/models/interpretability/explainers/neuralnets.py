import pandas as pd
from typing import Optional
from PySide6.QtCharts import QValueAxis
from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainers.neuralnets import NeuralNetsExplainer
from src.gui.components.dialogs.models.interpretability.explainers.base import ExplainerBaseDialog
from src.gui.components.dialogs.models.interpretability.explainers.ensemble import _FeatureImportanceChart


class _ImportanceWeightsChart(_FeatureImportanceChart):
    """ Feature-importance chart labeled for neural attention weights. """

    def set_importance_weights(self, importance_weights: pd.DataFrame, ascending: bool = False,):
        self.set_feature_importances(feature_importances=importance_weights, ascending=ascending)
        for axis in self._chart.axes():
            if isinstance(axis, QValueAxis):
                axis.setTitleText('Importance Weight')
        self._chart.setTitle(f'Importance Weights of {len(self._feature_names)} Best Features')


class NeuralNetsExplainerDialog(ExplainerBaseDialog):
    """Dialog for interpreting a stored neural-network model."""

    def __init__(self, df: pd.DataFrame, model_id: str, model_database: ModelDatabase, parent: Optional[QWidget] = None):
        self._importance_weights = None
        self._importance_sort_combo = None
        self._importance_chart = None

        explainer = NeuralNetsExplainer(df=df, model_id=model_id, database=model_database)
        self._has_attention_weights = explainer.has_attention_weights

        super().__init__(explainer=explainer, parent=parent)
        self.setWindowTitle(model_id)

    def explainer_model(self):
        super().explainer_model()

        if self._has_attention_weights:
            self._importance_weights = self._explainer.get_feature_importances()

    def build_analysis_frames(self):
        if self._importance_weights is not None:
            self._build_importance_weights_frame()

    def _build_importance_weights_frame(self):
        importance_frame = QGroupBox('Importance Weights', self)
        importance_frame.setObjectName('neuralNetworkImportanceWeightsFrame')
        self._set_analysis_frame_size(importance_frame)
        importance_layout = QVBoxLayout(importance_frame)
        importance_layout.setContentsMargins(12, 12, 12, 12)
        importance_layout.setSpacing(8)

        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel('Sort By Importance', importance_frame))
        self._importance_sort_combo = QComboBox(importance_frame)
        self._importance_sort_combo.addItems(['Descending', 'Ascending'])
        options_layout.addWidget(self._importance_sort_combo)
        options_layout.addStretch(1)
        importance_layout.addLayout(options_layout)

        self._importance_chart = _ImportanceWeightsChart(importance_frame)
        importance_layout.addWidget(self._importance_chart, 1)
        self._analysis_layout.addWidget(importance_frame, 1)

        self._importance_sort_combo.currentTextChanged.connect(self._update_importance_weights_chart)
        self._update_importance_weights_chart()

    def _update_importance_weights_chart(self, _value=None):
        self._importance_chart.set_importance_weights(
            importance_weights=self._importance_weights,
            ascending=self._importance_sort_combo.currentText() == 'Ascending',
        )
