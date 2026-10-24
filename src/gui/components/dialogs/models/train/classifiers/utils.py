import threading
import pandas as pd
from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QValueAxis
from PySide6.QtCore import QModelIndex, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget
)
from src.core.models.dashboard import (
    get_study_filepath,
    get_storage_url,
    start_optuna_dashboard,
    stop_optuna_dashboard,
    wait_optuna_server
)
from src.core.models.model import ModelConfig
from src.core.metrics.utils import CLASSIFICATION_METRICS, PROBABILITY_METRICS, SKILL_METRICS
from src.gui.animations.barplot import BarplotScaling
from src.gui.widgets import ExcelTable, PrimaryQPushButton

_MODEL_METRICS = tuple(dict.fromkeys(CLASSIFICATION_METRICS + PROBABILITY_METRICS + SKILL_METRICS))


class _ReportTabBar(QTabBar):
    """ Tab bar whose overflow arrows stay vertically centered. """

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


class _AverageRowDelegate(QStyledItemDelegate):
    """Render cross-validation average rows in red."""

    def initStyleOption(self, option, index: QModelIndex) -> None:
        super().initStyleOption(option, index)

        label = index.siblingAtColumn(0).data()
        if not isinstance(label, str) or not label.startswith('Average'):
            return

        color = QColor('red')
        option.palette.setColor(QPalette.ColorRole.Text, color)
        option.palette.setColor(QPalette.ColorRole.HighlightedText, color)


class _ImprovementRowDelegate(QStyledItemDelegate):
    """Color sliding-CV summary labels and signed metric changes."""

    def initStyleOption(self, option, index: QModelIndex) -> None:
        super().initStyleOption(option, index)

        label = index.siblingAtColumn(0).data()
        if not isinstance(label, str) or not label.startswith('Average Improvement'):
            return

        color = None
        if index.column() == 0:
            color = QColor('blue')
        elif index.column() >= 5:
            value = index.data()
            if isinstance(value, str) and value.startswith('+'):
                color = QColor('green')
            elif isinstance(value, str) and value.startswith('-') and value != '-':
                color = QColor('red')

        if color is not None:
            option.palette.setColor(QPalette.ColorRole.Text, color)
            option.palette.setColor(QPalette.ColorRole.HighlightedText, color)


class _NestedAverageRowDelegate(QStyledItemDelegate):
    """Color nested-CV averages by their role in model selection."""

    def initStyleOption(self, option, index: QModelIndex) -> None:
        super().initStyleOption(option, index)

        label = index.siblingAtColumn(0).data()
        if label != 'Average':
            return

        value = index.data()
        column_name = index.model().headerData(
            index.column(),
            Qt.Orientation.Horizontal,
            Qt.ItemDataRole.DisplayRole,
        )
        best_inner_prefix = 'Best Inner '
        best_inner_columns = [
            column
            for column in range(index.model().columnCount())
            if str(index.model().headerData(column, Qt.Orientation.Horizontal)).startswith(best_inner_prefix)
        ]
        optimized_metrics = {
            str(index.model().headerData(column, Qt.Orientation.Horizontal))[len(best_inner_prefix):]
            for column in best_inner_columns
        }

        if value == '-':
            color = QColor('black')
        elif str(column_name).startswith(best_inner_prefix) or column_name in optimized_metrics:
            color = QColor('red')
        else:
            color = QColor('blue')

        option.palette.setColor(QPalette.ColorRole.Text, color)
        option.palette.setColor(QPalette.ColorRole.HighlightedText, color)


class TrainEvalReportDialog(QDialog):
    """ Dialog displaying training, evaluation and tuning information. """

    def __init__(
            self, parent: QWidget, model_config: ModelConfig):
        super().__init__(parent)

        self._model_config = model_config

        self._dashboard_process = None
        self._dashboard_open_button = None
        self._dashboard_close_button = None
        self._barplot = []

        self._report_tabs = None

        self._setup_dialog()
        self._setup_ui()

    def _setup_dialog(self) -> None:
        """ Configure the dialog. """

        self.setWindowTitle(f'{self._model_config.model_id} - Training-Evaluation Report')
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
        """ Create the dialog interface. """

        self._report_tabs = QTabWidget(self)
        self._report_tabs.setTabBar(_ReportTabBar(self._report_tabs))

        tab_bar = self._report_tabs.tabBar()
        tab_bar.setFont(self._increase_font(tab_bar.font(), bold=True))
        tab_bar.setMinimumHeight(44)
        tab_bar.setExpanding(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self._report_tabs)

        self._add_reports()
        self._add_parameter_importances()

    def _add_reports(self) -> None:
        """ Add all available model reports. """

        self._add_table('Train-Eval Report', self._model_config.train_report)

        if self._model_config.cv_report is not None:
            cv_report = self._add_cross_validation_averages(self._model_config.cv_report)
            self._add_table(title='Cross Validation', dataframe=cv_report, color_average=True)
        if self._model_config.sliding_cv_report is not None:
            sliding_report = self._add_sliding_average_improvements(self._model_config.sliding_cv_report)
            self._add_table(
                title='Sliding Cross Validation',
                dataframe=sliding_report,
                color_improvement=True,
            )
        if self._model_config.tune_report is not None:
            self._add_tuning_report()
        if self._model_config.nested_cv_tune_report is not None:
            nested_report = self._add_nested_average(self._model_config.nested_cv_tune_report)
            self._add_table(
                title='Nested Cross Validation',
                dataframe=nested_report,
                color_nested_average=True,
            )

    def _add_table(
            self,
            title: str,
            dataframe: pd.DataFrame,
            color_average: bool = False,
            color_improvement: bool = False,
            color_nested_average: bool = False,
    ):
        """ Add a dataframe report as a new tab. """

        panel = QWidget(self._report_tabs)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 10, 0, 0)

        table = ExcelTable(
            parent=panel,
            df=dataframe,
            readonly=True,
            supports_sorting=False,
            supports_query_search=True,
            supports_deletion=False,
            auto_resize=True
        )
        if color_average:
            table.setItemDelegate(_AverageRowDelegate(table))
        elif color_improvement:
            table.setItemDelegate(_ImprovementRowDelegate(table))
        elif color_nested_average:
            table.setItemDelegate(_NestedAverageRowDelegate(table))

        layout.addWidget(table)
        self._report_tabs.addTab(panel, title)

    def _add_tuning_report(self) -> None:
        """Add the tuning report and Optuna dashboard controls."""

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
            df=self._model_config.tune_report,
            readonly=True,
            supports_sorting=False,
            supports_query_search=True,
            supports_deletion=False,
            auto_resize=True
        )
        layout.addWidget(table)

        self._dashboard_open_button.clicked.connect(self._open_dashboard)
        self._dashboard_close_button.clicked.connect(self._close_dashboard)
        self._report_tabs.addTab(panel, 'Tuning Report')

    def _add_nested_average(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Append averages for every nested-CV evaluation metric."""

        average_row = {column: '-' for column in dataframe.columns}
        average_row[dataframe.columns[0]] = 'Average'

        best_inner_columns = [column for column in dataframe.columns if str(column).startswith('Best Inner ')]
        declared_metric_columns = [metric for metric in _MODEL_METRICS if metric in dataframe.columns]

        # Model reports currently include AUC/AUC Skill even though those names
        # are not part of the exported metric constants. Include any such
        # report metrics while excluding tuned classifier parameters.
        parameter_columns = set(self._model_config.classifier_config)
        additional_metric_columns = [
            column
            for column in dataframe.columns
            if column != dataframe.columns[0]
            and column not in best_inner_columns
            and column not in declared_metric_columns
            and column not in parameter_columns
        ]
        metric_columns = list(dict.fromkeys(best_inner_columns + declared_metric_columns + additional_metric_columns))

        for column in metric_columns:
            values = pd.to_numeric(dataframe[column], errors='coerce')
            average = values.mean()
            average_row[column] = '-' if pd.isna(average) else round(float(average), 3)

        return pd.concat([dataframe, pd.DataFrame([average_row])], ignore_index=True)

    @staticmethod
    def _add_cross_validation_averages(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Append separate Train and Eval averages for all CV metric columns."""

        metric_columns = dataframe.columns[5:]
        summary_rows = []

        for data_type in ('Train', 'Eval'):
            data_rows = dataframe.loc[dataframe['Data'] == data_type, metric_columns]
            if data_rows.empty:
                continue

            average_row = {column: '-' for column in dataframe.columns}
            average_row[dataframe.columns[0]] = f'Average {data_type}'

            averages = data_rows.apply(pd.to_numeric, errors='coerce').mean()
            for column, value in averages.items():
                average_row[column] = '-' if pd.isna(value) else round(float(value), 3)

            summary_rows.append(average_row)

        if not summary_rows:
            return dataframe.copy()

        return pd.concat([dataframe, pd.DataFrame(summary_rows)], ignore_index=True)

    @staticmethod
    def _add_sliding_average_improvements(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Append mean consecutive metric changes for Train and Eval sliding-CV rows."""

        metric_columns = dataframe.columns[5:]
        summary_rows = []

        for data_type in ('Train', 'Eval'):
            data_rows = dataframe.loc[dataframe['Data'] == data_type, metric_columns]
            if data_rows.empty:
                continue

            improvement_row = {column: '-' for column in dataframe.columns}
            improvement_row[dataframe.columns[0]] = f'Average Improvement {data_type}'

            numeric_metrics = data_rows.apply(pd.to_numeric, errors='coerce')
            average_improvements = numeric_metrics.diff().mean()
            for column, value in average_improvements.items():
                if pd.isna(value):
                    improvement_row[column] = '-'
                    continue

                rounded_value = round(float(value), 3)
                if rounded_value == 0:
                    rounded_value = 0.0
                improvement_row[column] = f'{rounded_value:+.3f}'

            summary_rows.append(improvement_row)

        if not summary_rows:
            return dataframe.copy()

        return pd.concat([dataframe, pd.DataFrame(summary_rows)], ignore_index=True)

    def _add_parameter_importances(self) -> None:
        """ Add the Optuna parameter-importance barplot. """

        importances = self._model_config.tune_param_importances

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
        self._report_tabs.addTab(chart_view, 'Parameter Importances')
        animation = BarplotScaling(bar_set=bar_set, values=values, parent=chart_view)
        self._barplot.append(animation)
        animation.start()

    def _open_dashboard(self) -> None:
        """ Start the Optuna dashboard process. """

        self._dashboard_open_button.setEnabled(False)
        self._dashboard_close_button.setEnabled(True)

        try:
            storage_url = get_storage_url(study_filepath=get_study_filepath(league_id=self._model_config.league_id, entity='models'))
            self._dashboard_process = start_optuna_dashboard(storage_url=storage_url)
        except Exception as error:
            self._dashboard_process = None
            self._dashboard_close_button.setEnabled(False)
            self._dashboard_open_button.setEnabled(True)

            QMessageBox.critical(
                self,
                'Dashboard Unavailable',
                f'The Optuna dashboard for model "{self._model_config.model_id}" could not be started: {error}'
            )
            return
        else:
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
