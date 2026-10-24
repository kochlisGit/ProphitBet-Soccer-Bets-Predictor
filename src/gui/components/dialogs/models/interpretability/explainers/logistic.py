import numpy as np
import pandas as pd
from typing import List, Optional
from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCursor, QPainter
from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QSizePolicy, QToolTip, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainers.logistic import LogisticRegressionExplainer
from src.gui.animations.barplot import BarplotScaling
from src.gui.components.dialogs.models.interpretability.explainers.base import ExplainerBaseDialog, _increase_font


class _CoefficientChart(QChartView):
    """Horizontal chart of the strongest logistic-regression coefficients."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._chart = QChart()
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self._chart.legend().setFont(_increase_font(self._chart.legend().font()))
        self.setChart(self._chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._feature_names: List[str] = []
        self._hover_values = np.array([], dtype=float)
        self._barplot_scalings: List[BarplotScaling] = []

    def _reset(self):
        for animation in self._barplot_scalings:
            animation.stop()
            animation.deleteLater()
        self._barplot_scalings = []
        self._feature_names = []
        self._hover_values = np.array([], dtype=float)

        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)
            axis.deleteLater()
        self._chart.legend().hide()
        self._chart.setTitle('')

    @staticmethod
    def _chart_data(
            coefficients: pd.DataFrame,
            target=None,
    ) -> tuple[List[str], str, np.ndarray]:
        if {'Feature', 'Coefficient'}.issubset(coefficients.columns):
            features = coefficients['Feature'].astype(str).tolist()
            values = pd.to_numeric(coefficients['Coefficient'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            return features, 'Coefficient', values

        numeric_coefficients = coefficients.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        features = numeric_coefficients.columns.astype(str).tolist()
        if target is None:
            target = numeric_coefficients.index[0]
        values = numeric_coefficients.loc[target].to_numpy(dtype=float)
        return features, str(target), values

    def set_coefficients(self, coefficients: pd.DataFrame, target=None, ascending: bool = False):
        self._reset()

        feature_names, target_label, coefficient_values = self._chart_data(coefficients, target=target)
        axis_minimum = min(float(coefficient_values.min()), 0.0)
        axis_maximum = max(float(coefficient_values.max()), 0.0)
        if ascending:
            order = np.argsort(np.abs(coefficient_values), kind='stable').tolist()[:15]
        else:
            order = np.argsort(-np.abs(coefficient_values), kind='stable').tolist()[:15]

        feature_names = [feature_names[index] for index in order]
        coefficient_values = coefficient_values[order]
        self._feature_names = feature_names
        self._hover_values = coefficient_values

        bar_set = QBarSet(target_label)
        bar_set.append(coefficient_values.tolist())
        bar_set.setColor(QColor('#3D5AFE'))
        bar_set.setBorderColor(QColor('black'))

        series = QBarSeries()
        series.setBarWidth(0.65)
        series.append(bar_set)
        series.hovered.connect(self._show_tooltip)

        feature_axis = QBarCategoryAxis()
        feature_axis.append(self._feature_names)
        feature_axis.setTitleText('Feature')
        feature_axis.setLabelsAngle(90)
        feature_title_font = _increase_font(feature_axis.titleFont())
        feature_title_font.setBold(True)
        feature_axis.setTitleFont(feature_title_font)

        value_axis = QValueAxis()
        value_axis.setTitleText('Coefficient')
        value_title_font = _increase_font(value_axis.titleFont())
        value_title_font.setBold(True)
        value_axis.setTitleFont(value_title_font)
        value_axis.setLabelFormat('%.3f')
        padding = max((axis_maximum - axis_minimum) * 0.1, 0.01)
        value_axis.setRange(
            axis_minimum - padding if axis_minimum < 0.0 else 0.0,
            axis_maximum + padding,
        )
        value_axis.applyNiceNumbers()

        self._chart.addSeries(series)
        self._chart.addAxis(feature_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(value_axis, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(feature_axis)
        series.attachAxis(value_axis)

        title_font = _increase_font(feature_title_font)
        title_font.setBold(True)
        self._chart.setTitleFont(title_font)
        self._chart.setTitle(f'Coefficient Analysis of {len(self._feature_names)} Best Features')
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignTop)

        self._barplot_scalings = [
            BarplotScaling(bar_set, coefficient_values.tolist(), parent=self)
        ]
        for animation in self._barplot_scalings:
            animation.start()

    def _show_tooltip(self, status: bool, index: int, bar_set: QBarSet):
        if status and 0 <= index < len(self._feature_names):
            QToolTip.showText(
                QCursor.pos(),
                (
                    f'{self._feature_names[index]} - {bar_set.label()}: '
                    f'{self._hover_values[index]:.6f}'
                ),
                self,
            )
        else:
            QToolTip.hideText()


class LogisticRegressionExplainerDialog(ExplainerBaseDialog):
    """ Dialog for interpreting a stored logistic-regression model. """

    def __init__(self, df: pd.DataFrame, model_id: str, model_database: ModelDatabase, parent: Optional[QWidget] = None):
        self._coefficients = None
        self._coefficient_sort_combo = None
        self._coefficient_target_combo = None
        self._coefficient_chart = None

        explainer = LogisticRegressionExplainer(df=df, model_id=model_id, database=model_database)
        super().__init__(explainer=explainer, parent=parent)
        self.setWindowTitle(model_id)

    def explainer_model(self):
        super().explainer_model()

        self._coefficients = self._explainer.get_coefficients()

    def build_analysis_frames(self):
        self._build_coefficients_frame()

    def _build_coefficients_frame(self):
        coefficients_frame = QGroupBox('Coefficients Analysis', self)
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
