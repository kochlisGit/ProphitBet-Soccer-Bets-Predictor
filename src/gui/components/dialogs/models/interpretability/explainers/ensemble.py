import numpy as np
import pandas as pd
from typing import List, Optional
from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCursor, QPainter
from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QSizePolicy, QToolTip, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainers.tree import DecisionTreeExplainer
from src.gui.animations.barplot import BarplotScaling
from src.gui.components.dialogs.models.interpretability.explainers.base import ExplainerBaseDialog, _increase_font


class _FeatureImportanceChart(QChartView):
    """Horizontal chart of the strongest tree-based feature importances."""

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
            feature_importances: pd.DataFrame,
            target=None,
    ) -> tuple[List[str], str, np.ndarray]:
        if {'Feature', 'Importance'}.issubset(feature_importances.columns):
            features = feature_importances['Feature'].astype(str).tolist()
            values = pd.to_numeric(
                feature_importances['Importance'],
                errors='coerce',
            ).fillna(0.0).to_numpy(dtype=float)
            return features, 'Importance', values

        numeric_importances = feature_importances.apply(
            pd.to_numeric,
            errors='coerce',
        ).fillna(0.0)
        features = numeric_importances.columns.astype(str).tolist()
        if target is None:
            target = numeric_importances.index[0]
        values = numeric_importances.loc[target].to_numpy(dtype=float)
        return features, str(target), values

    def set_feature_importances(
            self,
            feature_importances: pd.DataFrame,
            target=None,
            ascending: bool = False,
    ):
        self._reset()

        feature_names, target_label, importance_values = self._chart_data(
            feature_importances,
            target=target,
        )
        axis_minimum = min(float(importance_values.min()), 0.0)
        axis_maximum = max(float(importance_values.max()), 0.0)
        if ascending:
            order = np.argsort(np.abs(importance_values), kind='stable').tolist()[:15]
        else:
            order = np.argsort(-np.abs(importance_values), kind='stable').tolist()[:15]

        feature_names = [feature_names[index] for index in order]
        importance_values = importance_values[order]
        self._feature_names = feature_names
        self._hover_values = importance_values

        bar_set = QBarSet(target_label)
        bar_set.append(importance_values.tolist())
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
        value_axis.setTitleText('Importance')
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
        self._chart.setTitle(
            f'Feature Importance Analysis of {len(self._feature_names)} Best Features'
        )
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignTop)

        self._barplot_scalings = [
            BarplotScaling(bar_set, importance_values.tolist(), parent=self)
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


class EnsembleRegressionExplainerDialog(ExplainerBaseDialog):
    """Dialog for interpreting a stored tree-based ensemble model."""

    def __init__(
            self,
            df: pd.DataFrame,
            model_id: str,
            model_database: ModelDatabase,
            parent: Optional[QWidget] = None,
    ):
        self._feature_importances = None
        self._importance_sort_combo = None
        self._importance_target_combo = None
        self._importance_chart = None

        explainer = DecisionTreeExplainer(
            df=df,
            model_id=model_id,
            database=model_database,
        )
        super().__init__(explainer=explainer, parent=parent)
        self.setWindowTitle(model_id)

    def explainer_model(self):
        super().explainer_model()

        self._feature_importances = self._explainer.get_feature_importances()

    def build_analysis_frames(self):
        self._build_feature_importances_frame()

    def _build_feature_importances_frame(self):
        importance_frame = QGroupBox('Feature Importances Analysis', self)
        self._set_analysis_frame_size(importance_frame)
        importance_layout = QVBoxLayout(importance_frame)
        importance_layout.setContentsMargins(12, 12, 12, 12)
        importance_layout.setSpacing(8)

        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel('Sort By Importance', importance_frame))
        self._importance_sort_combo = QComboBox(importance_frame)
        self._importance_sort_combo.addItems(['Descending', 'Ascending'])
        options_layout.addWidget(self._importance_sort_combo)

        if not {'Feature', 'Importance'}.issubset(self._feature_importances.columns):
            options_layout.addSpacing(16)
            options_layout.addWidget(QLabel('Target', importance_frame))
            self._importance_target_combo = QComboBox(importance_frame)
            for target in self._feature_importances.index:
                self._importance_target_combo.addItem(str(target), userData=target)
            options_layout.addWidget(self._importance_target_combo)

        options_layout.addStretch(1)
        importance_layout.addLayout(options_layout)

        self._importance_chart = _FeatureImportanceChart(importance_frame)
        importance_layout.addWidget(self._importance_chart, 1)
        self._analysis_layout.addWidget(importance_frame, 1)

        self._importance_sort_combo.currentTextChanged.connect(
            self._update_feature_importance_chart
        )
        if self._importance_target_combo is not None:
            self._importance_target_combo.currentIndexChanged.connect(
                self._update_feature_importance_chart
            )
        self._update_feature_importance_chart()

    def _update_feature_importance_chart(self, _value=None):
        target = (
            self._importance_target_combo.currentData()
            if self._importance_target_combo is not None
            else None
        )
        self._importance_chart.set_feature_importances(
            feature_importances=self._feature_importances,
            target=target,
            ascending=self._importance_sort_combo.currentText() == 'Ascending',
        )
