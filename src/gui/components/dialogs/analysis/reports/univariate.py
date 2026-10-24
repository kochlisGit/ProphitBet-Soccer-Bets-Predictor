import numpy as np
import pandas as pd
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple
from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QCategoryAxis,
    QChart,
    QChartView,
    QLineSeries,
    QValueAxis
)
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPointF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QFontMetrics, QPainter, QPaintEvent, QPalette, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionHeader,
    QStyleOptionViewItem,
    QTableView,
    QToolTip,
    QVBoxLayout,
    QWidget
)

from src.gui.animations.barplot import BarplotScaling


def _increase_font(font: QFont, amount: int = 2) -> QFont:
    """ Return a copy of a font increased by the requested amount. """

    font = QFont(font)
    if font.pointSize() > 0:
        font.setPointSize(font.pointSize() + amount)
    elif font.pixelSize() > 0:
        font.setPixelSize(font.pixelSize() + amount)
    return font


def _format_number(value: Any) -> str:
    """ Format report numbers compactly while preserving non-numeric category labels."""

    try:
        if pd.isna(value):
            return '—'
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f'{int(value):,}'
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return f'{int(number):,}'
        return f'{number:,.3f}'.rstrip('0').rstrip('.')
    return str(value)


def _format_x_value(value: Any) -> str:
    """ Format chart X values with no more than two decimal places. """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(number):
        return str(value)
    if number.is_integer():
        return f'{int(number):,}'
    return f'{number:,.2f}'.rstrip('0').rstrip('.')


class _StatisticsModel(QAbstractTableModel):
    """ Two-column view of the statistics belonging to one variable. """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rows: list[tuple[str, Any]] = []

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else 2

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        statistic, value = self._rows[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return statistic if index.column() == 0 else _format_number(value)
        if role == Qt.ItemDataRole.ToolTipRole and index.column() == 1:
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return ('Statistic', 'Value')[section]

        return super().headerData(section, orientation, role)

    def set_statistics(self, statistics: Optional[pd.Series]):
        self.beginResetModel()
        self._rows = [] if statistics is None else [(str(name), value) for name, value in statistics.items()]
        self.endResetModel()


class _StatisticsDelegate(QStyledItemDelegate):
    """ Paint table values with an eight-pixel left margin. """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        text = option.text
        option.text = ''

        option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget)

        font = QFont(option.font)
        font.setBold(index.column() == 0)
        painter.save()
        painter.setFont(font)
        painter.setPen(option.palette.color(QPalette.ColorRole.Text))
        painter.drawText(
            option.rect.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )
        painter.restore()


class _StatisticsHeader(QHeaderView):
    """ Left-align table headings with the same eight-pixel inset as their cells. """

    def paintSection(self, painter: QPainter, rect, logical_index: int):  # noqa: N802
        if not rect.isValid():
            return

        option = QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.section = logical_index
        self.style().drawControl(QStyle.ControlElement.CE_HeaderSection, option, painter, self)

        text = self.model().headerData(logical_index, self.orientation(), Qt.ItemDataRole.DisplayRole)
        painter.save()
        painter.setFont(self.font())
        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        painter.drawText(
            rect.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            str(text),
        )
        painter.restore()


class _Badge(QWidget):
    """ Small palette-aware badge used for issues and correlations. """

    def __init__(self, text: str, tone: str = 'issue', tooltip: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._text = text
        self._tone = tone

        font = self.font()
        font.setBold(True)
        self.setFont(font)
        self.setToolTip(tooltip or text)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: N802
        metrics = QFontMetrics(self.font())
        return QSize(metrics.horizontalAdvance(self._text) + 24, max(metrics.height() + 10, 28))

    def paintEvent(self, event: QPaintEvent):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#3949AB'))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)
        painter.setPen(self.palette().color(QPalette.ColorRole.ButtonText))
        painter.setFont(self.font())
        painter.drawText(self.rect().adjusted(12, 0, -12, 0), Qt.AlignmentFlag.AlignCenter, self._text)


class _DistributionChart(QChartView):
    """ Native Qt chart for a hoverable discrete bar plot or continuous histogram. """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._chart = QChart()
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        background = QBrush(QColor('#CFD8DC'))
        self._chart.setBackgroundBrush(background)
        self._chart.setPlotAreaBackgroundBrush(background)
        self._chart.setPlotAreaBackgroundVisible(True)
        self._chart.setTitleFont(_increase_font(self._chart.titleFont()))
        self._chart.legend().setFont(_increase_font(self._chart.legend().font()))
        self.setChart(self._chart)
        self.setBackgroundBrush(background)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._variable: Optional[str] = None
        self._distribution: Optional[Dict[str, Any]] = None
        self._bar_series = None
        self._bar_set = None
        self._line_series = None
        self._x_axis = None
        self._y_axis = None
        self._hover_messages: list[str] = []
        self._barplot_scaling = None
        self._draw_empty('Select a variable to view its distribution.')

    def _reset_chart(self):
        if self._barplot_scaling is not None:
            self._barplot_scaling.stop()
            self._barplot_scaling.deleteLater()
            self._barplot_scaling = None

        self._bar_series = None
        self._bar_set = None
        self._line_series = None
        self._x_axis = None
        self._y_axis = None
        self._hover_messages = []
        self._chart.removeAllSeries()

        for axis in self._chart.axes():
            self._chart.removeAxis(axis)
            axis.deleteLater()

        self._chart.legend().setVisible(False)
        self._chart.setTitle('')

    def _draw_empty(self, message: str):
        self._reset_chart()
        self._chart.setTitle(message)

    def set_distribution(self, variable: Any, distribution: Optional[Dict[str, Any]]):
        self._variable = str(variable)
        self._distribution = distribution
        self._reset_chart()

        if not distribution:
            self._draw_empty(f'No distribution is available for {self._variable}.')
            return

        distribution_type = str(distribution.get('Type', '')).lower()
        if distribution_type in {'discrete', 'bar', 'barplot'}:
            self._draw_barplot(distribution.get('Data'))
        elif distribution_type in {'continuous', 'histogram'}:
            self._draw_histogram(distribution.get('Data'))
        else:
            self._draw_empty(f'Unsupported distribution type for {self._variable}.')

    def _new_bar_set(self, label: str, counts: np.ndarray) -> QBarSet:
        bar_set = QBarSet(label)
        bar_set.append([float(count) for count in counts])
        bar_set.setColor(self.palette().color(QPalette.ColorRole.Highlight))
        bar_set.setBorderColor(QColor('black'))
        return bar_set

    @staticmethod
    def _configure_value_axis(axis: QValueAxis, title: str, maximum: float):
        axis.setTitleText(title)
        title_font = axis.titleFont()
        title_font.setBold(True)
        axis.setTitleFont(_increase_font(title_font))

        label_font = axis.labelsFont()
        label_font.setBold(False)
        axis.setLabelsFont(_increase_font(label_font))
        axis.setLabelFormat('%.0f')
        axis.setRange(0.0, max(float(maximum) * 1.1, 1.0))
        axis.applyNiceNumbers()

    def _draw_barplot(self, data: Any):
        if isinstance(data, pd.Series):
            categories = data.index.tolist()
            counts = data.to_numpy()
        elif isinstance(data, Mapping):
            categories = list(data.keys())
            counts = np.asarray(list(data.values()))
        else:
            categories, counts = [], np.asarray([])

        if not categories:
            self._draw_empty(f'No observations are available for {self._variable}.')
            return

        category_labels = [_format_x_value(category) for category in categories]
        self._hover_messages = [
            f'{_format_number(count)} observations have the value {category}.'
            for category, count in zip(category_labels, counts)
        ]

        self._bar_set = self._new_bar_set('', counts)
        self._bar_series = QBarSeries()
        self._bar_series.append(self._bar_set)
        self._bar_series.hovered.connect(self._show_hover_message)

        self._x_axis = QBarCategoryAxis()
        self._x_axis.setTitleText('Observation')
        x_title_font = self._x_axis.titleFont()
        x_title_font.setBold(True)
        self._x_axis.setTitleFont(_increase_font(x_title_font))
        self._x_axis.setLabelsFont(_increase_font(self._x_axis.labelsFont()))
        self._x_axis.append(category_labels)
        self._y_axis = QValueAxis()
        self._configure_value_axis(self._y_axis, 'Counts', np.max(counts))

        self._chart.addSeries(self._bar_series)
        self._bar_set.setColor(self.palette().color(QPalette.ColorRole.Highlight))
        self._bar_set.setBorderColor(QColor('black'))
        self._chart.addAxis(self._x_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self._y_axis, Qt.AlignmentFlag.AlignLeft)
        self._bar_series.attachAxis(self._x_axis)
        self._bar_series.attachAxis(self._y_axis)
        self._chart.setTitle(f'Barplot of {self._variable}')

        self._barplot_scaling = BarplotScaling(self._bar_set, counts, parent=self)
        self._barplot_scaling.start()

    def _draw_histogram(self, data: Any):
        if not isinstance(data, Mapping):
            self._draw_empty(f'No observations are available for {self._variable}.')
            return

        counts = np.asarray(data.get('counts', []), dtype=float)
        edges = np.asarray(data.get('bin_edges', []), dtype=float)
        centers = np.asarray(data.get('bin_centers', []), dtype=float)
        number_of_bins = min(len(counts), len(centers), max(len(edges) - 1, 0))

        if number_of_bins == 0:
            self._draw_empty(f'No observations are available for {self._variable}.')
            return

        counts = counts[:number_of_bins]
        centers = centers[:number_of_bins]
        edges = edges[:number_of_bins + 1]
        self._hover_messages = [
            (
                f'{_format_number(count)} observations fall between '
                f'[{_format_x_value(edges[index])}, {_format_x_value(edges[index + 1])}].'
            )
            for index, count in enumerate(counts)
        ]

        self._bar_set = self._new_bar_set('Observations', counts)
        self._bar_series = QBarSeries()
        self._bar_series.setBarWidth(1.0)
        self._bar_series.append(self._bar_set)
        self._bar_series.hovered.connect(self._show_hover_message)

        self._x_axis = QCategoryAxis()
        self._x_axis.setTitleText('Observation')
        x_title_font = self._x_axis.titleFont()
        x_title_font.setBold(True)
        self._x_axis.setTitleFont(_increase_font(x_title_font))
        self._x_axis.setLabelsFont(_increase_font(self._x_axis.labelsFont()))
        self._x_axis.setStartValue(-0.5)
        self._x_axis.setLabelsPosition(QCategoryAxis.AxisLabelsPosition.AxisLabelsPositionCenter)
        for index, center in enumerate(centers):
            self._x_axis.append(_format_x_value(center), index + 0.5)
        self._x_axis.setRange(-0.5, number_of_bins - 0.5)

        kde_x = np.asarray(data.get('kde_x', []), dtype=float)
        kde_y = np.asarray(data.get('kde_y', []), dtype=float)
        maximum = float(np.max(counts))
        if kde_x.size and kde_x.size == kde_y.size:
            maximum = max(maximum, float(np.max(kde_y)))

        self._y_axis = QValueAxis()
        self._configure_value_axis(self._y_axis, 'Counts', maximum)

        self._chart.addSeries(self._bar_series)
        self._bar_set.setColor(self.palette().color(QPalette.ColorRole.Highlight))
        self._bar_set.setBorderColor(QColor('black'))
        self._chart.addAxis(self._x_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self._y_axis, Qt.AlignmentFlag.AlignLeft)
        self._bar_series.attachAxis(self._x_axis)
        self._bar_series.attachAxis(self._y_axis)

        if kde_x.size and kde_x.size == kde_y.size:
            mapped_x = np.interp(kde_x, edges, np.arange(number_of_bins + 1) - 0.5)
            self._line_series = QLineSeries()
            self._line_series.setName('KDE')
            self._line_series.setPen(QPen(QColor('#e76f51'), 2))
            self._line_series.append([
                QPointF(float(x_value), float(y_value))
                for x_value, y_value in zip(mapped_x, kde_y)
            ])
            self._chart.addSeries(self._line_series)
            self._line_series.attachAxis(self._x_axis)
            self._line_series.attachAxis(self._y_axis)
            self._chart.legend().setVisible(True)

        self._chart.setTitle(f'Histogram of {self._variable}')

    def _show_hover_message(self, status: bool, index: int, bar_set: QBarSet):
        if status and 0 <= index < len(self._hover_messages):
            QToolTip.showText(QCursor.pos(), self._hover_messages[index], self)
        else:
            QToolTip.hideText()


class UnivariateReport(QFrame):
    """ Interactive report for one variable's statistics, alerts, and distribution.

        Attributes
        ----------
        :param descriptive_statistics: (pd.DataFrame) dataframe returned under Univariate['Descriptive Statistics'].
                                       Variables are rows and statistics are columns.
        :param distributions: (Dict[str, Any]) that describes the distributions of each variable via a Barplot or Hist.
        :param issues: (Dict[str, Any]) dict that maps an issue name to its description and the affected variables
                       paired with the metric value that triggered the issue.
        :param correlated_pairs: (List[Tuple[str, str, float]] | None) highly correlated variable pairs
                                 and their correlation values.
        :param imbalanced_variables: Optional pairs returned under ``Univariate['Imbalances']``.
        :param parent: (QWidget | None) parent widget.

    """

    def __init__(
            self,
            descriptive_statistics: pd.DataFrame,
            distributions: Dict[str, Dict[str, Any]],
            issues: Dict[str, Tuple[str, List[Tuple[str, float]]]],
            imbalanced_variables: List[Tuple[str, float]],
            correlated_pairs: Optional[List[Tuple[str, str, float]]] = None,
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._descriptive_statistics = descriptive_statistics.copy()
        self._distributions = distributions
        self._issues = issues
        self._imbalanced_variables = imbalanced_variables
        self._correlated_pairs = correlated_pairs or []

        self._issues_by_variable = self._index_issues()
        self._correlations_by_variable = self._index_correlations()
        self._imbalances_by_variable = dict(self._imbalanced_variables)

        self._configure_widget()
        self._populate_variables()

    @property
    def current_variable(self) -> Optional[str]:
        index = self.variable_combo.currentIndex()
        return None if index < 0 else self.variable_combo.itemData(index)

    def _configure_widget(self):
        self.setFont(_increase_font(self.font()))
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(900, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.statistics_panel = QFrame(self)
        self.statistics_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.distribution_panel = QFrame(self)
        self.distribution_panel.setFrameShape(QFrame.Shape.StyledPanel)

        self._configure_statistics_panel()
        self._configure_distribution_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.statistics_panel)
        splitter.addWidget(self.distribution_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([380, 570])

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)
        layout.addWidget(splitter)

    @staticmethod
    def _title_label(text: str, parent: QWidget) -> QLabel:
        label = QLabel(text, parent)
        font = label.font()
        font.setBold(True)
        if font.pointSize() > 0:
            font.setPointSize(font.pointSize() + 2)
        label.setFont(font)
        return label

    def _configure_statistics_panel(self):
        title = self._title_label('Descriptive Statistics', self.statistics_panel)
        variable_label = QLabel('Variable', self.statistics_panel)
        self.variable_combo = QComboBox(self.statistics_panel)
        self.variable_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        selector_layout = QHBoxLayout()
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(8)
        selector_layout.addWidget(variable_label)
        selector_layout.addWidget(self.variable_combo, 1)

        self.badges_content = QWidget(self.statistics_panel)
        self.badges_layout = QHBoxLayout(self.badges_content)
        self.badges_layout.setContentsMargins(0, 4, 0, 4)
        self.badges_layout.setSpacing(6)
        self.badges_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.badges_scroll = QScrollArea(self.statistics_panel)
        self.badges_scroll.setWidget(self.badges_content)
        self.badges_scroll.setWidgetResizable(True)
        self.badges_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.badges_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.badges_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.badges_scroll.setFixedHeight(54)

        self.statistics_model = _StatisticsModel(self)
        self.statistics_table = QTableView(self.statistics_panel)
        self.statistics_table.setModel(self.statistics_model)
        self.statistics_table.setItemDelegate(_StatisticsDelegate(self.statistics_table))
        self.statistics_table.setHorizontalHeader(
            _StatisticsHeader(Qt.Orientation.Horizontal, self.statistics_table)
        )
        self.statistics_table.setAlternatingRowColors(True)
        self.statistics_table.setShowGrid(False)
        self.statistics_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.statistics_table.setSelectionMode(QTableView.SelectionMode.NoSelection)
        self.statistics_table.verticalHeader().hide()
        self.statistics_table.verticalHeader().setDefaultSectionSize(34)
        header = self.statistics_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setMinimumSectionSize(110)

        layout = QVBoxLayout(self.statistics_panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addLayout(selector_layout)
        layout.addWidget(self.badges_scroll)
        layout.addWidget(self.statistics_table, 1)

        self.variable_combo.currentIndexChanged.connect(self._update_variable)

    def _configure_distribution_panel(self):
        title = self._title_label('Distribution', self.distribution_panel)
        self.distribution_plot = _DistributionChart(self.distribution_panel)

        layout = QVBoxLayout(self.distribution_panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(self.distribution_plot, 1)

    def _populate_variables(self):
        variables = list(self._descriptive_statistics.index)
        for variable in self._distributions:
            if variable not in variables:
                variables.append(variable)

        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()
        for variable in variables:
            self.variable_combo.addItem(str(variable), userData=variable)
        self.variable_combo.blockSignals(False)

        self.variable_combo.setEnabled(bool(variables))
        self._update_variable(0 if variables else -1)

    def set_variable(self, variable: str) -> bool:
        """Select a report variable. Returns whether the variable was found."""

        for index in range(self.variable_combo.count()):
            if self.variable_combo.itemData(index) == variable or self.variable_combo.itemText(index) == str(variable):
                self.variable_combo.setCurrentIndex(index)
                return True
        return False

    def _update_variable(self, index: int):
        if index < 0 or index >= self.variable_combo.count():
            self.statistics_model.set_statistics(None)
            self._update_badges(None)
            self.distribution_plot._draw_empty('No variables are available.')
            return

        variable = self.variable_combo.itemData(index)
        statistics = None
        if variable in self._descriptive_statistics.index:
            statistics = self._descriptive_statistics.loc[variable]

        self.statistics_model.set_statistics(statistics)
        self._update_badges(variable)
        self.distribution_plot.set_distribution(variable, self._distributions.get(variable))

    def _index_issues(self) -> dict[Any, list[tuple[str, str, float]]]:
        indexed = {}
        for issue, (description, variables) in self._issues.items():
            for variable, value in variables:
                indexed.setdefault(variable, []).append((issue, description, value))
        return indexed

    def _index_correlations(self) -> dict[Any, list[tuple[Any, float]]]:
        indexed = {}
        for first, second, correlation in self._correlated_pairs:
            indexed.setdefault(first, []).append((second, correlation))
            indexed.setdefault(second, []).append((first, correlation))
        return indexed

    def _clear_badges(self):
        while self.badges_layout.count():
            item = self.badges_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _update_badges(self, variable: Any):
        self._clear_badges()
        badges = []

        for issue, description, value in self._issues_by_variable.get(variable, []):
            value_text = _format_number(value)
            suffix = '%' if issue == 'High Missing' else ''
            badge = _Badge(
                f'{issue}: {value_text}{suffix}',
                tone='issue',
                tooltip=f'{variable} {description} Trigger value: {value_text}{suffix}',
                parent=self.badges_content,
            )
            badges.append(badge)

        if variable in self._imbalances_by_variable:
            entropy = _format_number(self._imbalances_by_variable[variable])
            badges.append(
                _Badge(
                    f'Imbalance: {entropy}',
                    tone='issue',
                    tooltip=f'{variable} has a normalized entropy score of {entropy}.',
                    parent=self.badges_content,
                )
            )

        for related_variable, correlation in self._correlations_by_variable.get(variable, []):
            correlation_text = _format_number(correlation)
            badges.append(
                _Badge(
                    f'Correlated: {related_variable} ({correlation_text})',
                    tone='correlation',
                    tooltip=f'{variable} and {related_variable} have a correlation of {correlation_text}.',
                    parent=self.badges_content,
                )
            )

        for badge in badges:
            self.badges_layout.addWidget(badge)

        self.badges_scroll.setVisible(bool(badges))
        self.badges_content.setMinimumWidth(self.badges_layout.sizeHint().width())
        self.badges_content.adjustSize()
