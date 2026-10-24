import threading
import pandas as pd
from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget
)
from src.core.agents.agent import AgentConfig
from src.core.models.dashboard import (
    get_study_filepath,
    get_storage_url,
    start_optuna_dashboard,
    stop_optuna_dashboard,
    wait_optuna_server
)
from src.gui.animations.barplot import BarplotScaling
from src.gui.widgets import ExcelTable, PrimaryQPushButton


class _ReportTabBar(QTabBar):
    """Tab bar whose overflow arrows stay vertically centered."""

    def _center_scroll_buttons(self) -> None:
        for button in self.findChildren(QToolButton):
            if button.arrowType() == Qt.ArrowType.NoArrow:
                continue

            button_height = min(button.sizeHint().height(), self.height())
            button.setGeometry(
                button.x(),
                (self.height() - button_height) // 2,
                button.width(),
                button_height,
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._center_scroll_buttons()

    def tabLayoutChange(self) -> None:
        super().tabLayoutChange()
        QTimer.singleShot(0, self._center_scroll_buttons)


class TrainEvalReportDialog(QDialog):
    """ Dialog displaying an agent's training, evaluation, and tuning reports. """

    def __init__(self, parent: QWidget, agent_config: AgentConfig):
        super().__init__(parent)

        self._agent_config = agent_config

        self._dashboard_process = None
        self._dashboard_open_button = None
        self._dashboard_close_button = None
        self._barplot = []
        self._report_tabs = None

        self._setup_dialog()
        self._setup_ui()

    def _setup_dialog(self) -> None:
        """Configure the dialog."""

        self.setWindowTitle(f'{self._agent_config.agent_id} - Training-Evaluation Report')
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setModal(False)
        self.setMinimumSize(600, 400)
        self.resize(800, 600)

    def _setup_ui(self) -> None:
        """Create the dialog interface."""

        self._report_tabs = QTabWidget(self)
        self._report_tabs.setTabBar(_ReportTabBar(self._report_tabs))

        tab_bar = self._report_tabs.tabBar()
        tab_bar.setFont(self._increase_font(tab_bar.font(), bold=True))
        tab_bar.setMinimumHeight(44)
        tab_bar.setExpanding(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self._report_tabs)

        self._add_train_eval_report()

        if isinstance(self._agent_config.tune_report, pd.DataFrame):
            self._add_tuning_report()

        self._add_parameter_importances()

    def _add_train_eval_report(self) -> None:
        """Add the reward-history plot and train/evaluation table."""

        panel = QWidget(self._report_tabs)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 10, 0, 0)

        chart_view = self._create_reward_history_chart(parent=panel)
        layout.addWidget(chart_view, 2)

        table = ExcelTable(
            parent=panel,
            df=self._agent_config.train_report,
            readonly=True,
            supports_sorting=False,
            supports_query_search=True,
            supports_deletion=False,
            auto_resize=True,
        )
        layout.addWidget(table, 1)
        self._report_tabs.addTab(panel, 'Train-Eval Report')

    def _create_reward_history_chart(self, parent: QWidget) -> QChartView:
        """Create a line chart showing average train and evaluation rewards."""

        dataframe = self._agent_config.train_report
        steps = pd.to_numeric(dataframe['Train Step'], errors='coerce')

        chart = QChart()
        chart.setTitle('Train and Evaluation Reward History')
        chart.setTitleFont(self._increase_font(chart.titleFont(), amount=2, bold=True))

        x_values = []
        y_values = []
        series_definitions = (
            ('Avg. Train Rewards', 'Train Rewards', QColor('#0d6efd')),
            ('Avg. Eval Rewards', 'Evaluation Rewards', QColor('#dc3545')),
        )

        for column, label, color in series_definitions:
            if column not in dataframe.columns:
                continue

            rewards = pd.to_numeric(dataframe[column], errors='coerce')
            line = QLineSeries()
            line.setName(label)
            line.setPen(QPen(color, 2))

            for step, reward in zip(steps, rewards):
                if pd.isna(step) or pd.isna(reward):
                    continue

                x_value = float(step)
                y_value = float(reward)
                line.append(x_value, y_value)
                x_values.append(x_value)
                y_values.append(y_value)

            chart.addSeries(line)

        x_axis = QValueAxis()
        x_axis.setTitleText('Training Step')
        x_axis.setTitleFont(self._increase_font(x_axis.titleFont(), amount=2, bold=True))
        x_axis.setLabelFormat('%.0f')

        y_axis = QValueAxis()
        y_axis.setTitleText('Average Reward')
        y_axis.setTitleFont(self._increase_font(y_axis.titleFont(), amount=2, bold=True))
        y_axis.setLabelFormat('%.2f')

        if x_values:
            x_min = min(x_values)
            x_max = max(x_values)
            if x_min == x_max:
                x_min = 0.0 if x_min > 0 else x_min - 1.0
                x_max += 1.0
        else:
            x_min, x_max = 0.0, 1.0
        x_axis.setRange(x_min, x_max)

        if y_values:
            # Keep zero inside the visible range so the reward baseline is
            # available even when every recorded reward has the same sign.
            y_min = min(min(y_values), 0.0)
            y_max = max(max(y_values), 0.0)
            padding = max((y_max - y_min) * 0.1, 0.1)
            y_axis.setRange(y_min - padding, y_max + padding)
        else:
            y_axis.setRange(-0.1, 0.1)

        zero_line = QLineSeries()
        zero_line.setName('Zero Baseline')
        zero_pen = QPen(QColor('black'), 1)
        zero_pen.setStyle(Qt.PenStyle.DashLine)
        zero_line.setPen(zero_pen)
        zero_line.append(x_min, 0.0)
        zero_line.append(x_max, 0.0)
        chart.addSeries(zero_line)

        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        for series in chart.series():
            series.attachAxis(x_axis)
            series.attachAxis(y_axis)

        for marker in chart.legend().markers(zero_line):
            marker.setVisible(False)

        chart_view = QChartView(chart, parent)
        chart_view.setMinimumHeight(260)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        return chart_view

    def _add_tuning_report(self) -> None:
        """ Add the tuning report and Optuna dashboard controls. """

        panel = QWidget(self._report_tabs)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 10, 0, 0)

        button_layout = QHBoxLayout()
        self._dashboard_open_button = PrimaryQPushButton(text='Open Optuna Dashboard', parent=panel)
        self._dashboard_close_button = QPushButton(text='Stop Optuna Dashboard', parent=panel)

        for button in (self._dashboard_open_button, self._dashboard_close_button):
            button.setFont(self._increase_font(button.font(), bold=True))
            button.setMinimumSize(230, 48)
            button_layout.addWidget(button)

        self._dashboard_close_button.setEnabled(False)
        button_layout.addStretch(1)
        layout.addLayout(button_layout)

        table = ExcelTable(
            parent=panel,
            df=self._agent_config.tune_report,
            readonly=True,
            supports_sorting=False,
            supports_query_search=True,
            supports_deletion=False,
            auto_resize=True,
        )
        layout.addWidget(table)

        self._dashboard_open_button.clicked.connect(self._open_dashboard)
        self._dashboard_close_button.clicked.connect(self._close_dashboard)
        self._report_tabs.addTab(panel, 'Tuning Report')

    def _add_parameter_importances(self) -> None:
        """ Add the Optuna hyperparameter-importance bar plot. """

        importances = self._agent_config.tune_param_importances

        if not importances:
            return

        values = [float(value) for value in importances.values()]

        bar_set = QBarSet('Importance')
        bar_set.append(values)
        bar_set.setColor(QColor('#0d6efd'))
        bar_set.setBorderColor(QColor('black'))

        series = QBarSeries()
        series.append(bar_set)
        series.setBarWidth(0.55)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle('Hyperparameter Importances')
        chart.setTitleFont(self._increase_font(chart.titleFont(), bold=True))
        chart.legend().setVisible(False)

        category_axis = QBarCategoryAxis()
        category_axis.append([str(parameter) for parameter in importances])
        category_axis.setTitleText('Hyperparameter')
        category_axis.setTitleFont(self._increase_font(category_axis.titleFont(), bold=True))
        category_axis.setLabelsFont(self._increase_font(category_axis.labelsFont(), bold=True))

        value_axis = QValueAxis()
        value_axis.setTitleText('Importance')
        value_axis.setTitleFont(self._increase_font(value_axis.titleFont(), bold=True))
        value_axis.setLabelsFont(self._increase_font(value_axis.labelsFont(), bold=True))
        value_axis.setRange(0.0, max(max(values) * 1.1, 1.0))
        value_axis.setLabelFormat('%.2f')

        chart.addAxis(category_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(value_axis, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(category_axis)
        series.attachAxis(value_axis)

        chart_view = QChartView(chart, self._report_tabs)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._report_tabs.addTab(chart_view, 'Hyperparameter Importances')

        animation = BarplotScaling(bar_set=bar_set, values=values, parent=chart_view)
        self._barplot.append(animation)
        animation.start()

    def _open_dashboard(self) -> None:
        """Start the agents Optuna dashboard process."""

        self._dashboard_open_button.setEnabled(False)
        self._dashboard_close_button.setEnabled(True)

        try:
            storage_url = get_storage_url(
                study_filepath=get_study_filepath(
                    league_id=self._agent_config.league_id,
                    entity='agents',
                )
            )
            self._dashboard_process = start_optuna_dashboard(storage_url=storage_url)
        except Exception as error:
            self._dashboard_process = None
            self._dashboard_close_button.setEnabled(False)
            self._dashboard_open_button.setEnabled(True)

            QMessageBox.critical(
                self,
                'Dashboard Unavailable',
                f'The Optuna dashboard for agent "{self._agent_config.agent_id}" could not be started: {error}',
            )
            return

        threading.Thread(target=wait_optuna_server, args=(), daemon=True).start()

    def _close_dashboard(self) -> None:
        """Stop the Optuna dashboard process."""

        if self._dashboard_process is None:
            if self._dashboard_close_button is not None:
                self._dashboard_close_button.setEnabled(False)

            if self._dashboard_open_button is not None:
                self._dashboard_open_button.setEnabled(True)

            return

        try:
            stop_optuna_dashboard(process=self._dashboard_process)
        finally:
            self._dashboard_process = None
            self._dashboard_close_button.setEnabled(False)
            self._dashboard_open_button.setEnabled(True)

    def done(self, result: int) -> None:
        """Close the dialog and stop any active dashboard."""

        self._close_dashboard()
        super().done(result)

    @staticmethod
    def _increase_font(font, amount: int = 2, bold: bool = False):
        """Increase a font's size."""

        if font.pointSize() > 0:
            font.setPointSize(font.pointSize() + amount)
        elif font.pixelSize() > 0:
            font.setPixelSize(font.pixelSize() + amount)
        else:
            font.setPointSize(12)

        if bold:
            font.setBold(True)

        return font
