from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QValueAxis
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTabBar,
    QToolTip,
    QVBoxLayout,
    QWidget
)

from src.gui.animations.barplot import BarplotScaling


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
    return f'{float(value):,.2f}'.rstrip('0').rstrip('.')


class _FeatureSelectionChart(QChartView):
    """Full-width native Qt bar chart for feature-selection scores."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._chart = QChart()
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self._chart.legend().hide()
        self._chart.setTitleFont(_increase_font(self._chart.titleFont()))

        background = QBrush(QColor('#CFD8DC'))
        self._chart.setBackgroundBrush(background)
        self._chart.setPlotAreaBackgroundBrush(background)
        self._chart.setPlotAreaBackgroundVisible(True)

        self.setChart(self._chart)
        self.setBackgroundBrush(background)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        self._hover_messages: list[str] = []
        self._barplot_scaling = None
        self._draw_empty('No feature-selection results are available.')

    def _reset_chart(self):
        if self._barplot_scaling is not None:
            self._barplot_scaling.stop()
            self._barplot_scaling.deleteLater()
            self._barplot_scaling = None

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

    def set_scores(
            self,
            dataframe: Optional[pd.DataFrame],
            value_column: str,
            title: str,
            y_axis_title: str,
            maximum: Optional[float] = None
    ):
        self._reset_chart()
        if (
                dataframe is None
                or dataframe.empty
                or 'Feature' not in dataframe.columns
                or value_column not in dataframe.columns
        ):
            self._draw_empty(f'No {title.lower()} results are available.')
            return

        values = pd.to_numeric(dataframe[value_column], errors='coerce')
        valid = values.notna()
        if not valid.any():
            self._draw_empty(f'No {title.lower()} results are available.')
            return

        features = dataframe.loc[valid, 'Feature'].astype(str).tolist()
        scores = values.loc[valid].to_numpy(dtype=float)
        features = features[:25]
        scores = scores[:25]

        bar_set = QBarSet(y_axis_title)
        bar_set.append(scores.tolist())
        bar_set.setColor(self.palette().color(QPalette.ColorRole.Highlight))
        bar_set.setBorderColor(QColor('black'))

        series = QBarSeries()
        series.setBarWidth(0.5)
        series.append(bar_set)
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
        y_axis.setTitleText(y_axis_title)
        y_axis.setLabelFormat('%.2f')
        score_maximum = float(np.max(scores))
        axis_maximum = min(score_maximum + 0.1, maximum) if maximum is not None else score_maximum + 0.1
        y_axis.setRange(0.0, max(axis_maximum, 0.1))

        if maximum is None:
            y_axis.applyNiceNumbers()
            
        y_axis.setLabelsFont(_increase_font(y_axis.labelsFont()))
        y_title_font = _increase_font(y_axis.titleFont())
        y_title_font.setBold(True)
        y_axis.setTitleFont(y_title_font)

        self._chart.addSeries(series)
        bar_set.setBorderColor(QColor('black'))
        self._chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)
        self._chart.setTitle(title)

        self._hover_messages = [
            f'{feature}: {_format_number(score)}'
            for feature, score in zip(features, scores)
        ]

        self._barplot_scaling = BarplotScaling(bar_set, scores, parent=self)
        self._barplot_scaling.start()

    def _show_hover_message(self, status: bool, index: int, bar_set: QBarSet):
        if status and 0 <= index < len(self._hover_messages):
            QToolTip.showText(
                QCursor.pos(),
                self._hover_messages[index],
                self
            )
        else:
            QToolTip.hideText()


class FeatureSelectionReport(QFrame):
    """Report for normalized variance scores and target-specific Boruta ranks.

    Parameters
    ----------
    normalized_variance:
        Dataframe returned under ``Selection['Normalized Variance']``.
    boruta:
        Target-to-dataframe mapping returned under ``Selection['Boruta']``.
        A single dataframe is also accepted.
    parent:
        Optional parent widget.
    """

    def __init__(
            self,
            normalized_variance: pd.DataFrame,
            boruta: Union[Dict[str, pd.DataFrame], pd.DataFrame],
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._normalized_variance = normalized_variance
        self._boruta = boruta

        self._configure_widget()
        self._populate_targets()
        self._update_chart()

    def _configure_widget(self):
        self.setFont(_increase_font(self.font()))
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(900, 520)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        title = QLabel('Feature Selection', self)
        title_font = _increase_font(title.font())
        title_font.setBold(True)
        title.setFont(title_font)

        self.mode_tabs = QTabBar(self)
        self.mode_tabs.setExpanding(False)
        self.mode_tabs.setDrawBase(False)
        self.variance_index = self.mode_tabs.addTab('Normalized Variance')
        self.boruta_index = self.mode_tabs.addTab('Boruta Rankings')
        tab_spacing = QWidget(self.mode_tabs)
        tab_spacing.setFixedWidth(12)
        tab_spacing.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.mode_tabs.setTabButton(
            self.variance_index,
            QTabBar.ButtonPosition.RightSide,
            tab_spacing
        )

        self.target_label = QLabel('Target:', self)
        self.target_combo = QComboBox(self)
        self.target_combo.setMinimumWidth(150)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addWidget(self.mode_tabs)
        toolbar.addStretch(1)
        toolbar.addWidget(self.target_label)
        toolbar.addWidget(self.target_combo)

        self.plot = _FeatureSelectionChart(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addLayout(toolbar)
        layout.addWidget(self.plot, 1)

        self.mode_tabs.currentChanged.connect(self._mode_changed)
        self.target_combo.currentIndexChanged.connect(self._update_chart)

    def _populate_targets(self):
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        for target in self._boruta:
            self.target_combo.addItem(str(target), userData=target)
        self.target_combo.setCurrentIndex(0 if self._boruta else -1)
        self.target_combo.blockSignals(False)
        self.mode_tabs.setTabEnabled(self.boruta_index, bool(self._boruta))
        self._update_target_visibility()

    def _mode_changed(self, index: int):
        self._update_target_visibility()
        self._update_chart()

    def _update_target_visibility(self):
        visible = self.mode_tabs.currentIndex() == self.boruta_index
        self.target_label.setVisible(visible)
        self.target_combo.setVisible(visible)

    def _current_boruta(self) -> tuple[Any, Optional[pd.DataFrame]]:
        index = self.target_combo.currentIndex()
        if index < 0:
            return None, None
        target = self.target_combo.itemData(index)
        return target, self._boruta.get(target)

    def _update_chart(self, index: int = -1):
        if self.mode_tabs.currentIndex() == self.variance_index:
            self.plot.set_scores(
                dataframe=self._normalized_variance,
                value_column='Variance',
                title='Normalized Variance',
                y_axis_title='Variance',
                maximum=1.0
            )
            return

        target, dataframe = self._current_boruta()
        title = 'Boruta Rankings' if target is None else f'Boruta Rankings - {target}'
        self.plot.set_scores(
            dataframe=dataframe,
            value_column='Rank',
            title=title,
            y_axis_title='Rank'
        )
