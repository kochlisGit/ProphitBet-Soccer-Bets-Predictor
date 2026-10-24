import numpy as np
import pandas as pd
from collections.abc import Mapping
from typing import Any, Dict, Optional
from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPalette
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QSizePolicy, QTabBar, QToolTip, QVBoxLayout, QWidget
from src.gui.animations.barplot import BarplotScaling
from src.core.preprocessing.utils.targets import TargetType


def _increase_font(font: QFont, amount: int = 2) -> QFont:
    """Return a copy of a font increased by the requested amount."""

    font = QFont(font)
    if font.pointSize() > 0:
        font.setPointSize(font.pointSize() + amount)
    elif font.pixelSize() > 0:
        font.setPixelSize(font.pixelSize() + amount)
    return font


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return f'{int(value):,}'
    return f'{float(value):,.3f}'.rstrip('0').rstrip('.')


def _as_target_mapping(results: Dict[str, pd.DataFrame]) -> Dict[Any, pd.DataFrame]:
    """Normalize analyzer results while also accepting one standalone dataframe."""

    if isinstance(results, pd.DataFrame):
        return {'All': results}
    if isinstance(results, Mapping):
        return dict(results)
    return {}


class _FeatureImportanceChart(QChartView):
    """ Native Qt bar chart for one target's feature-importance scores. """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._chart = QChart()
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self._chart.setTitleFont(_increase_font(self._chart.titleFont()))

        background = QBrush(QColor('#CFD8DC'))
        self._chart.setBackgroundBrush(background)
        self._chart.setPlotAreaBackgroundBrush(background)
        self._chart.setPlotAreaBackgroundVisible(True)

        self.setChart(self._chart)
        self.setBackgroundBrush(background)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._hover_messages: list[list[str]] = []
        self._barplot_scalings: list[BarplotScaling] = []
        self._draw_empty('No feature-importance results are available.')

    def _reset_chart(self):
        for animation in self._barplot_scalings:
            animation.stop()
            animation.deleteLater()
        self._barplot_scalings = []
        self._hover_messages = []

        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)
            axis.deleteLater()
        self._chart.legend().hide()
        self._chart.setTitle('')

    def _draw_empty(self, message: str):
        self._reset_chart()
        self._chart.setTitle(message)

    @staticmethod
    def _chart_data(dataframe: pd.DataFrame, value_column: str) -> tuple[list[str], list[tuple[str, np.ndarray]]]:
        feature_column = next(
            (column for column in ('Feature', 'Features') if column in dataframe.columns),
            None,
        )

        if value_column in dataframe.columns:
            values = pd.to_numeric(dataframe[value_column], errors='coerce')
            valid = values.notna()
            if not valid.any():
                return [], []
            features = (
                dataframe.loc[valid, feature_column].astype(str).tolist()
                if feature_column is not None
                else dataframe.index[valid].astype(str).tolist()
            )
            return features, [(value_column, values.loc[valid].to_numpy(dtype=float))]

        if value_column != 'Coefficient':
            return [], []

        coefficient_data = dataframe.drop(columns=[feature_column], errors='ignore').apply(
            pd.to_numeric,
            errors='coerce',
        )
        valid_columns = [column for column in coefficient_data if coefficient_data[column].notna().any()]
        if not valid_columns:
            return [], []

        coefficient_data = coefficient_data[valid_columns].fillna(0.0)
        features = (
            dataframe[feature_column].astype(str).tolist()
            if feature_column is not None
            else dataframe.index.astype(str).tolist()
        )
        return features, [
            (str(column), coefficient_data[column].to_numpy(dtype=float))
            for column in valid_columns
        ]

    def set_scores(self, dataframe: Optional[pd.DataFrame], value_column: str, title: str):
        self._reset_chart()
        if dataframe is None or dataframe.empty:
            self._draw_empty(f'No {title.lower()} results are available.')
            return

        features, score_groups = self._chart_data(dataframe, value_column)
        if not features or not score_groups:
            self._draw_empty(f'No {title.lower()} results are available.')
            return

        features = features[:25]
        score_groups = [(label, scores[:25]) for label, scores in score_groups]

        series = QBarSeries()
        series.setBarWidth(0.5)
        maximum = 0.0
        for group_index, (label, scores) in enumerate(score_groups):
            bar_set = QBarSet(label)
            bar_set.append(scores.tolist())
            if len(score_groups) == 1:
                bar_set.setColor(self.palette().color(QPalette.ColorRole.Highlight))
            bar_set.setBorderColor(QColor('black'))
            series.append(bar_set)
            maximum = max(maximum, float(np.max(scores)))
            self._hover_messages.append([
                f'{feature} — {label}: {_format_number(score)}'
                for feature, score in zip(features, scores)
            ])
            animation = BarplotScaling(bar_set, scores, parent=self)
            self._barplot_scalings.append(animation)

        series.hovered.connect(self._show_hover_message)

        x_axis = QBarCategoryAxis()
        x_axis.append(features)
        x_axis.setTitleText('Feature')
        x_axis.setLabelsAngle(90)
        x_axis.setLabelsFont(_increase_font(x_axis.labelsFont()))
        x_title_font = _increase_font(x_axis.titleFont())
        x_title_font.setBold(True)
        x_axis.setTitleFont(x_title_font)

        y_axis = QValueAxis()
        y_axis.setTitleText(value_column)
        y_axis.setLabelFormat('%.2f')
        y_axis.setRange(0.0, max(maximum * 1.1, 0.1))
        y_axis.applyNiceNumbers()
        y_axis.setLabelsFont(_increase_font(y_axis.labelsFont()))
        y_title_font = _increase_font(y_axis.titleFont())
        y_title_font.setBold(True)
        y_axis.setTitleFont(y_title_font)

        self._chart.addSeries(series)
        for bar_set in series.barSets():
            bar_set.setBorderColor(QColor('black'))
        self._chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)
        self._chart.setTitle(title)
        self._chart.legend().setVisible(len(score_groups) > 1)

        for animation in self._barplot_scalings:
            animation.start()

    def _show_hover_message(self, status: bool, index: int, bar_set: QBarSet):
        sets = self._chart.series()[0].barSets() if self._chart.series() else []
        try:
            group_index = sets.index(bar_set)
        except ValueError:
            group_index = -1

        if status and 0 <= group_index < len(self._hover_messages) and 0 <= index < len(self._hover_messages[group_index]):
            QToolTip.showText(QCursor.pos(), self._hover_messages[group_index][index], self)
        else:
            QToolTip.hideText()


class FeatureImportanceReport(QFrame):
    """Report for target-specific linear coefficients and Gini impurity scores."""

    def __init__(
            self,
            linear_coefficients: Dict[str, pd.DataFrame],
            gini_impurity_scores: Dict[str, pd.DataFrame],
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._linear_coefficients = _as_target_mapping(linear_coefficients)
        self._gini_impurity_scores = _as_target_mapping(gini_impurity_scores)

        self.setFont(_increase_font(self.font()))
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(900, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        title = QLabel('Feature Importance', self)
        title_font = _increase_font(title.font())
        title_font.setBold(True)
        title.setFont(title_font)

        self.mode_tabs = QTabBar(self)
        self.mode_tabs.setExpanding(False)
        self.mode_tabs.setDrawBase(False)
        self.linear_index = self.mode_tabs.addTab('Linear Coefficients')
        self.gini_index = self.mode_tabs.addTab('Gini Impurity')
        tab_spacing = QWidget(self.mode_tabs)
        tab_spacing.setFixedWidth(12)
        tab_spacing.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.mode_tabs.setTabButton(
            self.linear_index,
            QTabBar.ButtonPosition.RightSide,
            tab_spacing
        )

        self.target_label = QLabel('Target:', self)
        self.target_combo = QComboBox(self)
        self.target_combo.setMinimumWidth(150)

        self.order_label = QLabel('Order By:', self)
        self.order_combo = QComboBox(self)
        self.order_combo.setMinimumWidth(100)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addWidget(self.mode_tabs)
        toolbar.addStretch(1)
        toolbar.addWidget(self.target_label)
        toolbar.addWidget(self.target_combo)
        toolbar.addWidget(self.order_label)
        toolbar.addWidget(self.order_combo)

        self.plot = _FeatureImportanceChart(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addLayout(toolbar)
        layout.addWidget(self.plot, 1)

        self._populate_targets()
        self.mode_tabs.currentChanged.connect(self._mode_changed)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.order_combo.currentIndexChanged.connect(self._update_chart)
        self._update_order_options()
        self._update_chart()

    def _populate_targets(self):
        targets = list(self._linear_coefficients)
        for target in self._gini_impurity_scores:
            if target not in targets:
                targets.append(target)

        self.target_combo.blockSignals(True)
        for target in targets:
            self.target_combo.addItem(str(target), userData=target)
        self.target_combo.setCurrentIndex(0 if targets else -1)
        self.target_combo.setEnabled(bool(targets))
        self.target_combo.blockSignals(False)

        self.mode_tabs.setTabEnabled(self.linear_index, bool(self._linear_coefficients))
        self.mode_tabs.setTabEnabled(self.gini_index, bool(self._gini_impurity_scores))

    def _current_target(self) -> Any:
        index = self.target_combo.currentIndex()
        return self.target_combo.itemData(index) if index >= 0 else None

    @staticmethod
    def _target_classes(target: Any) -> list[str]:
        if target is None:
            return []
        try:
            mapper = TargetType(target).get_mapper()
        except (TypeError, ValueError, NotImplementedError):
            return []
        return list(mapper.keys())

    def _mode_changed(self, index: int):
        self._update_order_options()
        self._update_chart()

    def _target_changed(self, index: int):
        self._update_order_options()
        self._update_chart()

    def _update_order_options(self):
        target = self._current_target()
        classes = self._target_classes(target)
        dataframe = self._linear_coefficients.get(target)
        available_classes = (
            [class_name for class_name in classes if class_name in dataframe.columns]
            if dataframe is not None
            else []
        )
        is_multiclass_linear = (
            self.mode_tabs.currentIndex() == self.linear_index
            and len(classes) > 2
            and dataframe is not None
        )

        previous_class = self.order_combo.currentData()
        self.order_combo.blockSignals(True)
        self.order_combo.clear()
        for class_name in available_classes:
            self.order_combo.addItem(str(class_name), userData=class_name)
        previous_index = self.order_combo.findData(previous_class)
        self.order_combo.setCurrentIndex(previous_index if previous_index >= 0 else (0 if available_classes else -1))
        self.order_combo.blockSignals(False)
        self.order_label.setVisible(is_multiclass_linear)
        self.order_combo.setVisible(is_multiclass_linear)

    def _update_chart(self, index: int = -1):
        target_index = self.target_combo.currentIndex()
        target = self.target_combo.itemData(target_index) if target_index >= 0 else None

        if self.mode_tabs.currentIndex() == self.linear_index:
            results = self._linear_coefficients
            title = 'Linear Coefficients'
            value_column = 'Coefficient'
        else:
            results = self._gini_impurity_scores
            title = 'Gini Impurity'
            value_column = 'Gini Impurity'

        dataframe = results.get(target) if target is not None else None
        if self.mode_tabs.currentIndex() == self.linear_index and dataframe is not None:
            classes = self._target_classes(target)
            order_class = self.order_combo.currentData()
            if len(classes) > 2 and order_class in dataframe.columns:
                dataframe = dataframe.sort_values(by=order_class, ascending=False)

        chart_title = title if target is None or target == 'All' else f'{title} - {target}'
        self.plot.set_scores(dataframe, value_column, chart_title)
