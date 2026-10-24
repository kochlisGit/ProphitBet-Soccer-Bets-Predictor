import pandas as pd
from collections.abc import Iterable
from functools import partial
from typing import Any, Optional
from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QCategoryAxis,
    QDateTimeAxis,
    QLineSeries,
    QScatterSeries,
    QValueAxis
)
from PySide6.QtCore import QDateTime, QEvent, QPointF, Signal, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QPainter,
    QPalette,
    QPen,
    QStandardItem,
    QStandardItemModel
)
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QSizePolicy, QTabBar, QToolTip, QVBoxLayout, QWidget)
from src.gui.animations.barplot import BarplotScaling


BACKGROUND = '#CFD8DC'
PERFORMANCE_MEASURES = ('Wins', 'Goals', 'Points')
TOTAL_PERFORMANCE_MEASURES = ('Wins', 'Goals', 'Points', 'GF', 'AG')


def _increase_font(font: QFont, amount: int = 2) -> QFont:
    """Return a copy of a font increased by the requested amount."""

    font = QFont(font)
    if font.pointSize() > 0:
        font.setPointSize(font.pointSize() + amount)
    elif font.pixelSize() > 0:
        font.setPixelSize(font.pixelSize() + amount)
    return font


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    """Return non-null values once, preserving their original order."""

    unique = []
    for value in values:
        if pd.isna(value) or value in unique:
            continue
        unique.append(value)
    return unique


class _TeamCheckableComboBox(QComboBox):
    """ Compact team multi-selector with checkable popup rows. """

    selectionChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText('Select teams')
        self.setModel(QStandardItemModel(self))
        self.view().viewport().installEventFilter(self)
        self.setMinimumWidth(110)

    def set_items(self, values: Iterable[Any], checked_values: Optional[Iterable[Any]] = None):
        values = list(values)
        checked_values = list(checked_values or [])
        model = self.model()
        model.clear()

        for value in values:
            item = QStandardItem(str(value))
            item.setData(value, Qt.ItemDataRole.UserRole)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            state = Qt.CheckState.Checked if value in checked_values else Qt.CheckState.Unchecked
            item.setData(state, Qt.ItemDataRole.CheckStateRole)
            model.appendRow(item)
        self._update_summary()

    def checked_values(self) -> list[Any]:
        values = []
        model = self.model()
        for row in range(model.rowCount()):
            item = model.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                values.append(item.data(Qt.ItemDataRole.UserRole))
        return values

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        if watched is self.view().viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            index = self.view().indexAt(event.position().toPoint())

            if index.isValid():
                item = self.model().itemFromIndex(index)
                checked = item.checkState() == Qt.CheckState.Checked
                item.setCheckState(Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked)
                self._update_summary()
                self.selectionChanged.emit()
                return True

        return super().eventFilter(watched, event)

    def _update_summary(self):
        count = len(self.checked_values())
        self.setEditText(f'{count} team' if count == 1 else f'{count} teams')


class _SeasonalPerformanceChart(QChartView):
    """ Native Qt line chart for cumulative team performance by season. """

    COLORS = (
        '#0d6efd',
        '#dc3545',
        '#198754',
        '#fd7e14',
        '#6f42c1',
        '#20c997',
        '#d63384',
        '#ffc107',
        '#0dcaf0',
        '#795548',
    )

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._chart = QChart()
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self._chart.setBackgroundBrush(QBrush(QColor(BACKGROUND)))
        self._chart.setPlotAreaBackgroundBrush(QBrush(QColor(BACKGROUND)))
        self._chart.setPlotAreaBackgroundVisible(True)
        title_font = _increase_font(self._chart.titleFont())
        title_font.setBold(True)
        self._chart.setTitleFont(title_font)

        self.setChart(self._chart)
        self.setBackgroundBrush(QBrush(QColor(BACKGROUND)))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._barplot_scalings: list[BarplotScaling] = []
        self._bar_hover_messages: list[list[str]] = []
        self.show_message('Select at least one team to display seasonal performance.')

    def _reset(self):
        for animation in self._barplot_scalings:
            animation.stop()
            animation.deleteLater()
        self._barplot_scalings = []
        self._bar_hover_messages = []
        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)
            axis.deleteLater()
        self._chart.legend().hide()
        self._chart.setTitle('')

    def show_message(self, message: str):
        self._reset()
        self._chart.setTitle(message)

    @classmethod
    def _team_color(cls, team_index: int) -> QColor:
        if team_index < len(cls.COLORS):
            return QColor(cls.COLORS[team_index])
        return QColor.fromHsv((team_index * 137) % 360, 180, 210)

    @staticmethod
    def _configure_axis_fonts(axis):
        labels_font = _increase_font(axis.labelsFont())
        axis.setLabelsFont(labels_font)
        title_font = _increase_font(axis.titleFont())
        title_font.setBold(True)
        axis.setTitleFont(title_font)

    @staticmethod
    def _format_value(value: float) -> str:
        return (
            f'{int(value):,}'
            if float(value).is_integer()
            else f'{value:,.2f}'.rstrip('0').rstrip('.')
        )

    def _show_seasonal_tooltip(
            self,
            team: Any,
            measure: str,
            weeks: dict[float, Any],
            point: QPointF,
            state: bool,
    ):
        if state:
            week = weeks.get(point.x())
            week_text = (
                f'Week {self._format_value(float(week))}: '
                if week is not None and not pd.isna(week)
                else ''
            )
            QToolTip.showText(
                QCursor.pos(),
                f'{week_text}{team}: {self._format_value(point.y())} {measure}',
                self,
            )
        else:
            QToolTip.hideText()

    def _show_total_tooltip(
            self,
            team: Any,
            measure: str,
            seasons: dict[float, Any],
            point: QPointF,
            state: bool,
    ):
        if state:
            QToolTip.showText(
                QCursor.pos(),
                (
                    f'Season: {seasons.get(point.x())}, '
                    f'Team: {team}, '
                    f'{measure}: {self._format_value(point.y())}'
                ),
                self,
            )
        else:
            QToolTip.hideText()

    def _show_bar_tooltip(self, state: bool, index: int, bar_set: QBarSet):
        series = self._chart.series()
        sets = series[0].barSets() if series and isinstance(series[0], QBarSeries) else []
        try:
            set_index = sets.index(bar_set)
        except ValueError:
            set_index = -1

        if (
                state
                and 0 <= set_index < len(self._bar_hover_messages)
                and 0 <= index < len(self._bar_hover_messages[set_index])
        ):
            QToolTip.showText(QCursor.pos(), self._bar_hover_messages[set_index][index], self)
        else:
            QToolTip.hideText()

    def set_performance(
            self,
            dataframe: pd.DataFrame,
            teams: list[Any],
            measure: str,
            season: Any,
            team_indexes: dict[Any, int],
    ):
        self._reset()
        required = {'Date', 'Season', 'Week', 'Team', measure}

        if dataframe.empty or not required.issubset(dataframe.columns):
            self.show_message('No seasonal-performance results are available.')
            return

        if not teams:
            self.show_message('Select at least one team to display seasonal performance.')
            return

        selected = dataframe.loc[dataframe['Season'].eq(season) & dataframe['Team'].isin(teams), ['Date', 'Week', 'Team', measure]].copy()
        selected['Date'] = pd.to_datetime(selected['Date'], errors='coerce')
        selected[measure] = pd.to_numeric(selected[measure], errors='coerce')
        selected = selected.dropna(subset=['Date', measure]).sort_values(by=['Date', 'Team'], kind='stable')
        plotted_teams = [team for team in teams if selected['Team'].eq(team).any()]

        if not plotted_teams:
            self.show_message('No seasonal-performance results are available.')
            return

        minimum_date = None
        maximum_date = None
        maximum_value = 0.0
        marker_series = []
        for team in plotted_teams:
            team_data = selected[selected['Team'].eq(team)]
            color = self._team_color(team_indexes.get(team, 0))

            line = QLineSeries()
            line.setName(str(team))
            line.setPen(QPen(color, 2.0))

            markers = QScatterSeries()
            markers.setName(str(team))
            markers.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeCircle)
            markers.setMarkerSize(9.0)
            markers.setColor(color)
            markers.setBorderColor(QColor('black'))
            weeks = {
                float(int(date.value // 1_000_000)): week
                for date, week in zip(team_data['Date'], team_data['Week'])
            }
            markers.hovered.connect(
                partial(
                    self._show_seasonal_tooltip,
                    team,
                    measure,
                    weeks,
                )
            )

            for date, value in zip(team_data['Date'], team_data[measure]):
                milliseconds = int(date.value // 1_000_000)
                point = QPointF(float(milliseconds), float(value))
                line.append(point)
                markers.append(point)
                minimum_date = milliseconds if minimum_date is None else min(minimum_date, milliseconds)
                maximum_date = milliseconds if maximum_date is None else max(maximum_date, milliseconds)
                maximum_value = max(maximum_value, float(value))

            self._chart.addSeries(line)
            self._chart.addSeries(markers)
            marker_series.append(markers)

        date_axis = QDateTimeAxis()
        date_axis.setTitleText('Date')
        date_axis.setFormat('dd MMM')
        date_axis.setTickCount(8)
        self._configure_axis_fonts(date_axis)

        if minimum_date == maximum_date:
            one_day = 24 * 60 * 60 * 1000
            minimum_date -= one_day
            maximum_date += one_day
        date_axis.setRange(QDateTime.fromMSecsSinceEpoch(minimum_date), QDateTime.fromMSecsSinceEpoch(maximum_date))

        value_axis = QValueAxis()
        value_axis.setTitleText(measure)
        value_axis.setLabelFormat('%d')
        value_axis.setRange(0.0, max(maximum_value * 1.1, 1.0))
        value_axis.applyNiceNumbers()
        self._configure_axis_fonts(value_axis)

        self._chart.addAxis(date_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(value_axis, Qt.AlignmentFlag.AlignLeft)
        for series in self._chart.series():
            series.attachAxis(date_axis)
            series.attachAxis(value_axis)

        for markers in marker_series:
            legend_markers = self._chart.legend().markers(markers)
            if legend_markers:
                legend_markers[0].setVisible(False)

        title = (
            f'{measure} of {plotted_teams[0]} in {season}'
            if len(plotted_teams) == 1
            else f'{measure} of {len(plotted_teams)} teams in {season}'
        )
        self._chart.setTitle(title)
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

    def set_total_performance(
            self,
            dataframe: pd.DataFrame,
            teams: list[Any],
            measure: str,
            team_indexes: dict[Any, int],
    ):
        """ Plot each selected team's seasonal totals across all seasons. """

        self._reset()
        required = {'Season', 'Team', measure}

        if dataframe.empty or not required.issubset(dataframe.columns):
            self.show_message('No total-performance results are available.')
            return

        if not teams:
            self.show_message('Select at least one team to display total performance.')
            return

        seasons = _ordered_unique(dataframe['Season'].tolist())
        try:
            seasons = sorted(seasons)
        except TypeError:
            pass
        season_indexes = {
            season: index for index, season in enumerate(seasons)
        }

        selected = dataframe.loc[dataframe['Team'].isin(teams), ['Season', 'Team', measure]].copy()
        selected[measure] = pd.to_numeric(selected[measure], errors='coerce')
        selected = selected.dropna(subset=['Season', measure])
        selected['_Season Index'] = selected['Season'].map(season_indexes)
        selected = selected.sort_values(by=['_Season Index', 'Team'], kind='stable')
        plotted_teams = [team for team in teams if selected['Team'].eq(team).any()]

        if not plotted_teams or not seasons:
            self.show_message('No total-performance results are available.')
            return

        maximum_value = 0.0
        marker_series = []
        for team in plotted_teams:
            team_data = selected[selected['Team'].eq(team)]
            color = self._team_color(team_indexes.get(team, 0))

            line = QLineSeries()
            line.setName(str(team))
            line.setPen(QPen(color, 2.0))

            markers = QScatterSeries()
            markers.setName(str(team))
            markers.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeCircle)
            markers.setMarkerSize(9.0)
            markers.setColor(color)
            markers.setBorderColor(QColor('black'))
            point_seasons = {float(season_indexes[season]): season for season in team_data['Season']}
            markers.hovered.connect(
                partial(
                    self._show_total_tooltip,
                    team,
                    measure,
                    point_seasons,
                )
            )

            for season, value in zip(team_data['Season'], team_data[measure]):
                point = QPointF(float(season_indexes[season]), float(value))
                line.append(point)
                markers.append(point)
                maximum_value = max(maximum_value, float(value))

            self._chart.addSeries(line)
            self._chart.addSeries(markers)
            marker_series.append(markers)

        season_axis = QCategoryAxis()
        season_axis.setTitleText('Season')
        season_axis.setStartValue(-0.5)
        season_axis.setLabelsPosition(
            QCategoryAxis.AxisLabelsPosition.AxisLabelsPositionCenter
        )
        season_axis.setLabelsAngle(-45)
        for index, season in enumerate(seasons):
            season_axis.append(str(season), index + 0.5)
        season_axis.setRange(-0.5, len(seasons) - 0.5)
        self._configure_axis_fonts(season_axis)

        value_axis = QValueAxis()
        value_axis.setTitleText(measure)
        value_axis.setLabelFormat('%.2f' if measure in ('GF', 'AG') else '%d')
        value_axis.setRange(0.0, max(maximum_value * 1.1, 1.0))
        value_axis.applyNiceNumbers()
        self._configure_axis_fonts(value_axis)

        self._chart.addAxis(season_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(value_axis, Qt.AlignmentFlag.AlignLeft)

        for series in self._chart.series():
            series.attachAxis(season_axis)
            series.attachAxis(value_axis)

        for markers in marker_series:
            legend_markers = self._chart.legend().markers(markers)
            if legend_markers:
                legend_markers[0].setVisible(False)

        title = (
            f'{measure} of {plotted_teams[0]} across seasons'
            if len(plotted_teams) == 1
            else f'{measure} of {len(plotted_teams)} teams across seasons'
        )
        self._chart.setTitle(title)
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)


    def set_predictability(self, dataframe: pd.DataFrame, team: str):
        """Plot actual and expected wins, draws and losses for one team."""

        self._reset()
        categories = ('Wins', 'Draws', 'Losses')
        actual_columns = tuple(f'Total {category}' for category in categories)
        expected_columns = tuple(f'Total Expected {category}' for category in categories)
        required = {'Team', *actual_columns, *expected_columns}

        if dataframe.empty or not required.issubset(dataframe.columns):
            self.show_message('No predictability results are available.')
            return

        if team is None:
            self.show_message('Select a team to display predictability.')
            return

        team_rows = dataframe[dataframe['Team'].eq(team)]

        if team_rows.empty:
            self.show_message('No predictability results are available.')
            return

        row = team_rows.iloc[0]
        actual = pd.to_numeric(row[list(actual_columns)], errors='coerce')
        expected = pd.to_numeric(row[list(expected_columns)], errors='coerce')

        if actual.isna().any() or expected.isna().any():
            self.show_message('No predictability results are available.')
            return

        actual_values = actual.to_numpy(dtype=float).tolist()
        expected_values = expected.to_numpy(dtype=float).tolist()

        actual_set = QBarSet('Total')
        actual_set.append(actual_values)
        actual_set.setColor(self.palette().color(QPalette.ColorRole.Highlight))
        actual_set.setBorderColor(QColor('black'))

        expected_set = QBarSet('Total Expected')
        expected_set.append(expected_values)
        expected_set.setColor(QColor('#FF1744'))
        expected_set.setBorderColor(QColor('black'))

        series = QBarSeries()
        series.setBarWidth(0.65)
        series.append(actual_set)
        series.append(expected_set)
        series.hovered.connect(self._show_bar_tooltip)
        self._chart.addSeries(series)
        actual_set.setBorderColor(QColor('black'))
        expected_set.setBorderColor(QColor('black'))

        category_axis = QBarCategoryAxis()
        category_axis.append(list(categories))
        category_axis.setTitleText('Result')
        self._configure_axis_fonts(category_axis)

        value_axis = QValueAxis()
        value_axis.setTitleText('Total')
        value_axis.setLabelFormat('%d')
        maximum = max(actual_values + expected_values, default=0.0)
        value_axis.setRange(0.0, max(maximum * 1.1, 1.0))
        value_axis.applyNiceNumbers()
        self._configure_axis_fonts(value_axis)

        self._chart.addAxis(category_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(value_axis, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(category_axis)
        series.attachAxis(value_axis)

        self._bar_hover_messages = [
            [
                f'{team}, {category}: {self._format_value(value)}'
                for category, value in zip(categories, values)
            ]
            for values in (actual_values, expected_values)
        ]
        self._chart.setTitle(f'Actual vs Expected Results of {team}')
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

        self._barplot_scalings = [
            BarplotScaling(actual_set, actual_values, parent=self),
            BarplotScaling(expected_set, expected_values, parent=self),
        ]
        for animation in self._barplot_scalings:
            animation.start()


class TeamPerformanceReport(QFrame):
    """Interactive report for timeline-based team performance."""

    def __init__(
            self,
            timeline: pd.DataFrame,
            seasonal_timeline: pd.DataFrame,
            actual_expected_results: pd.DataFrame,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._timeline = timeline.copy() if isinstance(timeline, pd.DataFrame) else pd.DataFrame()
        self._seasonal_timeline = (
            seasonal_timeline.copy()
            if isinstance(seasonal_timeline, pd.DataFrame)
            else pd.DataFrame()
        )
        self._actual_expected_results = (
            actual_expected_results.copy()
            if isinstance(actual_expected_results, pd.DataFrame)
            else pd.DataFrame()
        )
        self._all_teams = self._collect_teams()
        self._team_indexes = {team: index for index, team in enumerate(self._all_teams)}

        self.setFont(_increase_font(self.font()))
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(900, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        title = QLabel('Team Performance', self)
        title_font = _increase_font(title.font())
        title_font.setBold(True)
        title.setFont(title_font)

        self.mode_tabs = QTabBar(self)
        self.mode_tabs.setExpanding(False)
        self.mode_tabs.setDrawBase(False)
        self.seasonal_index = self.mode_tabs.addTab('Seasonal Performance')
        self.total_index = self.mode_tabs.addTab('Total Performance')
        self.predictability_index = self.mode_tabs.addTab('Predictability')

        self.team_label = QLabel('Teams:', self)
        self.teams_combo = _TeamCheckableComboBox(self)

        self.team_combo = QComboBox(self)
        self.team_combo.setMinimumWidth(130)
        for team in self._all_teams:
            self.team_combo.addItem(str(team), userData=team)

        self.measure_label = QLabel('Measure:', self)
        self.measure_combo = QComboBox(self)
        self.measure_combo.addItems(PERFORMANCE_MEASURES)
        self.measure_combo.setMinimumWidth(90)

        self.season_label = QLabel('Season:', self)
        self.season_combo = QComboBox(self)
        self.season_combo.setMinimumWidth(85)
        for season in self._seasons():
            self.season_combo.addItem(str(season), userData=season)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addWidget(self.mode_tabs)
        toolbar.addStretch(1)
        toolbar.addWidget(self.team_label)
        toolbar.addWidget(self.teams_combo)
        toolbar.addWidget(self.team_combo)
        toolbar.addWidget(self.measure_label)
        toolbar.addWidget(self.measure_combo)
        toolbar.addWidget(self.season_label)
        toolbar.addWidget(self.season_combo)

        self.plot = _SeasonalPerformanceChart(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addLayout(toolbar)
        layout.addWidget(self.plot, 1)

        self._populate_multi_teams()
        self.mode_tabs.currentChanged.connect(self._mode_changed)
        self.teams_combo.selectionChanged.connect(self._update_plot)
        self.team_combo.currentIndexChanged.connect(self._update_plot)
        self.measure_combo.currentIndexChanged.connect(self._update_plot)
        self.season_combo.currentIndexChanged.connect(self._season_changed)
        self._update_control_visibility()
        self._update_plot()

    def _collect_teams(self) -> list[Any]:
        teams = []
        for dataframe in (self._timeline, self._seasonal_timeline, self._actual_expected_results):
            if 'Team' in dataframe.columns:
                teams.extend(dataframe['Team'].tolist())
        return _ordered_unique(teams)

    def _seasons(self) -> list[Any]:
        values = []
        for dataframe in (self._timeline, self._seasonal_timeline):
            if 'Season' in dataframe.columns:
                values.extend(dataframe['Season'].tolist())
        seasons = _ordered_unique(values)
        try:
            return sorted(seasons, reverse=True)
        except TypeError:
            return seasons

    def _teams_for_current_season(self) -> list[Any]:
        if not {'Season', 'Team'}.issubset(self._timeline.columns):
            return []
        season = self.season_combo.currentData()
        return _ordered_unique(
            self._timeline.loc[
                self._timeline['Season'].eq(season),
                'Team',
            ].tolist()
        )

    def _populate_multi_teams(self):
        if self.mode_tabs.currentIndex() == self.total_index:
            available = (
                _ordered_unique(self._seasonal_timeline['Team'].tolist())
                if 'Team' in self._seasonal_timeline.columns
                else []
            )
        else:
            available = self._teams_for_current_season()
        previous = self.teams_combo.checked_values()
        checked = [team for team in previous if team in available]
        if not checked and available:
            checked = available[:1]
        self.teams_combo.set_items(available, checked_values=checked)
        self.teams_combo.setEnabled(bool(available))

    def _mode_changed(self, index: int):
        self._update_measure_options()
        if index != self.predictability_index:
            self._populate_multi_teams()
        self._update_control_visibility()
        self._update_plot()

    def _season_changed(self, index: int):
        if self.mode_tabs.currentIndex() == self.seasonal_index:
            self._populate_multi_teams()
        self._update_plot()

    def _update_measure_options(self):
        measures = (
            TOTAL_PERFORMANCE_MEASURES
            if self.mode_tabs.currentIndex() == self.total_index
            else PERFORMANCE_MEASURES
        )
        previous = self.measure_combo.currentText()
        self.measure_combo.blockSignals(True)
        self.measure_combo.clear()
        self.measure_combo.addItems(measures)
        previous_index = self.measure_combo.findText(previous)
        self.measure_combo.setCurrentIndex(
            previous_index if previous_index >= 0 else 0
        )
        self.measure_combo.blockSignals(False)

    def _update_control_visibility(self):
        mode = self.mode_tabs.currentIndex()
        multi_team = mode in (self.seasonal_index, self.total_index)
        seasonal = mode == self.seasonal_index
        predictability = mode == self.predictability_index

        self.team_label.setText('Team:' if predictability else 'Teams:')
        self.teams_combo.setVisible(multi_team)
        self.team_combo.setVisible(predictability)
        self.measure_label.setVisible(multi_team)
        self.measure_combo.setVisible(multi_team)
        self.season_label.setVisible(seasonal)
        self.season_combo.setVisible(seasonal)

    def _update_plot(self, index: int = -1):
        mode = self.mode_tabs.currentIndex()
        if mode == self.predictability_index:
            self.plot.set_predictability(
                dataframe=self._actual_expected_results,
                team=self.team_combo.currentData(),
            )
            return

        if mode == self.seasonal_index:
            self.plot.set_performance(
                dataframe=self._timeline,
                teams=self.teams_combo.checked_values(),
                measure=self.measure_combo.currentText(),
                season=self.season_combo.currentData(),
                team_indexes=self._team_indexes,
            )
        else:
            self.plot.set_total_performance(
                dataframe=self._seasonal_timeline,
                teams=self.teams_combo.checked_values(),
                measure=self.measure_combo.currentText(),
                team_indexes=self._team_indexes,
            )
