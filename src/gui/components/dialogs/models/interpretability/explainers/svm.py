import pandas as pd
from typing import Optional
from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainers.svm import SupportVectorMachineExplainer
from src.gui.components.dialogs.models.interpretability.explainers.base import ExplainerBaseDialog
from src.gui.components.dialogs.models.interpretability.explainers.logistic import _CoefficientChart


class SupportVectorMachineExplainerDialog(ExplainerBaseDialog):
    """Dialog for interpreting a stored support-vector machine model."""

    def __init__(self, df: pd.DataFrame, model_id: str, model_database: ModelDatabase, parent: Optional[QWidget] = None):
        self._coefficients = None
        self._coefficient_sort_combo = None
        self._coefficient_target_combo = None
        self._coefficient_chart = None

        explainer = SupportVectorMachineExplainer(df=df, model_id=model_id, database=model_database)
        kernel = explainer.model_config.classifier_config.get('kernel')
        self._supports_coefficients = kernel is None or kernel == 'linear'

        super().__init__(explainer=explainer, parent=parent)
        self.setWindowTitle(model_id)

    def explainer_model(self):
        super().explainer_model()

        if self._supports_coefficients:
            self._coefficients = self._explainer.get_coefficients()

    def build_analysis_frames(self):
        if self._supports_coefficients:
            self._build_coefficients_frame()

    def _build_coefficients_frame(self):
        coefficients_frame = QGroupBox('Coefficients Analysis', self)
        coefficients_frame.setObjectName('svmCoefficientsFrame')
        self._set_analysis_frame_size(coefficients_frame)
        coefficients_layout = QVBoxLayout(coefficients_frame)
        coefficients_layout.setContentsMargins(12, 12, 12, 12)
        coefficients_layout.setSpacing(8)

        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel('Sort By Importance', coefficients_frame))
        self._coefficient_sort_combo = QComboBox(coefficients_frame)
        self._coefficient_sort_combo.addItems(['Descending', 'Ascending'])
        options_layout.addWidget(self._coefficient_sort_combo)

        if not {'Feature', 'Coefficient'}.issubset(self._coefficients.columns):
            options_layout.addSpacing(16)
            options_layout.addWidget(QLabel('Target', coefficients_frame))
            self._coefficient_target_combo = QComboBox(coefficients_frame)
            for target in self._coefficients.index:
                self._coefficient_target_combo.addItem(str(target), userData=target)
            options_layout.addWidget(self._coefficient_target_combo)

        options_layout.addStretch(1)
        coefficients_layout.addLayout(options_layout)

        self._coefficient_chart = _CoefficientChart(coefficients_frame)
        coefficients_layout.addWidget(self._coefficient_chart, 1)
        self._analysis_layout.addWidget(coefficients_frame, 1)

        self._coefficient_sort_combo.currentTextChanged.connect(self._update_coefficient_chart)
        if self._coefficient_target_combo is not None:
            self._coefficient_target_combo.currentIndexChanged.connect(self._update_coefficient_chart)
        self._update_coefficient_chart()

    def _update_coefficient_chart(self, _value=None):
        target = (
            self._coefficient_target_combo.currentData()
            if self._coefficient_target_combo is not None
            else None
        )
        self._coefficient_chart.set_coefficients(
            coefficients=self._coefficients,
            target=target,
            ascending=self._coefficient_sort_combo.currentText() == 'Ascending',
        )
