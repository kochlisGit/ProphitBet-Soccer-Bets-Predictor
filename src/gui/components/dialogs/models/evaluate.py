import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Dict, List, Optional
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget
)
from src.core.agents.environments.actions import Action
from src.core.agents.utils import BETTING_PERFORMANCE_METRICS, compute_betting_metrics
from src.core.database.agent import AgentDatabase
from src.core.database.model import ModelDatabase
from src.core.metrics.utils import (
    CLASSIFICATION_METRICS,
    ODDS_COLUMNS,
    DOUBLE_CHANCE_ODD_COLUMNS,
    PROBABILITY_METRICS,
    SKILL_METRICS,
    generate_report,
    generate_report_dc
)
from src.core.preprocessing.filters import EvalFilter, filter_matches
from src.core.preprocessing.utils.targets import TargetType
from src.gui.components.dialogs.models.utils import EXTRA_METRIC_COLUMNS, YearlyEvaluationDialog
from src.gui.components.dialogs.tasks import Task
from src.gui.widgets import CheckableComboBox, ExcelTable, FloatDoubleRangeSlider

_MODEL_CONFIG_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ModelConfigWriter')
_ODD_FILTER_STEP = 0.1


def _expand_odd_slider_range(minimum: float, maximum: float) -> tuple[float, float]:
    """Expand data limits onto the slider's displayed 0.1 odds grid."""

    step = Decimal(str(_ODD_FILTER_STEP))
    lower = (Decimal(str(minimum)) / step).to_integral_value(rounding=ROUND_FLOOR) * step
    upper = (Decimal(str(maximum)) / step).to_integral_value(rounding=ROUND_CEILING) * step

    # The range slider requires two distinct endpoints, even if every match
    # happens to have the same odd.
    if upper <= lower:
        upper = lower + step

    return float(lower), float(upper)


def _load_agent_and_compute_actions(
        agent_id: str,
        agent_database: AgentDatabase,
        df: pd.DataFrame
) -> tuple[np.ndarray, float, Dict[str, float]]:
    """Load an agent and compute its actions inside a task process."""

    agent, _ = agent_database.load_agent(agent_id=agent_id)
    return agent.compute_actions(df=df)


class EvaluateModelDialog(QDialog):
    def __init__(
            self,
            df: pd.DataFrame,
            model_database: ModelDatabase,
            agent_database: AgentDatabase,
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._df = df.dropna(ignore_index=True)
        self._model_database = model_database
        self._agent_database = agent_database

        # Fetching model ids per target.
        self._model_ids = {target_type: [] for target_type in TargetType}
        league_models = self._model_database.index.get(self._model_database.league_id, {})
        for model_id, model_config in league_models.items():
            target_type = model_config.classifier_config['target_type']
            self._model_ids[target_type].append(model_id)

        # Constructing metrics table.
        metric_columns = CLASSIFICATION_METRICS + PROBABILITY_METRICS + SKILL_METRICS + BETTING_PERFORMANCE_METRICS
        self._metrics_df = pd.DataFrame([[np.nan] * len(metric_columns)], columns=metric_columns, dtype=object)

        # Extracting probability columns per each target.
        self._target_columns = {
            target_type: [
                f'Prob-{target_class}'
                for target_class in target_type.get_mapper()
            ]
            for target_type in TargetType
        }
        self._probability_columns = sum(self._target_columns.values(), [])
        for target_type in TargetType:
            self._target_columns[target_type].insert(0, target_type.value)

        # Computing odd min-max values.
        odd_lower_limits = self._df[ODDS_COLUMNS + DOUBLE_CHANCE_ODD_COLUMNS].min(axis=0)
        odd_upper_limits = self._df[ODDS_COLUMNS + DOUBLE_CHANCE_ODD_COLUMNS].max(axis=0)
        self._odd_limits = {
            odd: _expand_odd_slider_range(odd_lower_limits[odd], odd_upper_limits[odd])
            for odd in odd_lower_limits.index.tolist()
        }

        # Extracting targets.
        self._targets_dict = {target_type.value: self._df[target_type.value] for target_type in TargetType}

        # Declaring Placeholders.
        self._target_type = None
        self._model_config = None
        self._y_pred_ids = None
        self._y_pred = None
        self._y_prob = None
        self._y_pred_dc = None
        self._y_prob_dc = None
        self._recent_ids = None
        self._eval_ids = None
        self._train_ids = None
        self._correct_ids = None
        self._correct_dc_ids = None
        self._dataset_ids = None
        self._odd_percentile_filter_ids = None
        self._min_probability = 0.0
        self._actions = None
        self._total_betting_metrics = [np.nan]*len(BETTING_PERFORMANCE_METRICS)
        self._yearly_evaluation_dialog = None

        # Building GUI.
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setMinimumSize(600, 400)
        self.resize(800, 600)
        self.build_ui()
        self.setWindowTitle('Evaluate Models')

    def done(self, result: int):
        if self._yearly_evaluation_dialog is not None:
            self._yearly_evaluation_dialog.shutdown()
        super().done(result)

    def build_ui(self):
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        dialog_layout.addWidget(scroll_area)

        page = QWidget(scroll_area)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(16, 16, 16, 16)
        page_layout.setSpacing(12)

        selectors_group = QGroupBox('Selection', page)
        selectors_layout = QGridLayout(selectors_group)
        selectors_layout.setContentsMargins(12, 12, 12, 12)
        selectors_layout.setHorizontalSpacing(12)
        selectors_layout.setVerticalSpacing(10)

        selection_combo_width = 180

        target_label = QLabel('Target Type', selectors_group)
        self._target_combo = QComboBox(selectors_group)
        self._target_combo.setFixedWidth(selection_combo_width)
        for target_type, model_ids in self._model_ids.items():
            if model_ids:
                self._target_combo.addItem(target_type.value, userData=target_type)
        self._target_combo.setCurrentIndex(-1)
        self._target_combo.setToolTip('Selects models trained on the specified target variable.')
        selectors_layout.addWidget(target_label, 0, 0)
        selectors_layout.addWidget(self._target_combo, 0, 1)

        model_label = QLabel('Model ID', selectors_group)
        self._model_combo = QComboBox(selectors_group)
        self._model_combo.setFixedWidth(selection_combo_width * 2 - 60)
        self._model_combo.setToolTip('Unique model identifier.')
        selectors_layout.addWidget(model_label, 0, 3)
        selectors_layout.addWidget(self._model_combo, 0, 4)

        dataset_label = QLabel('Dataset', selectors_group)
        self._dataset_combo = QComboBox(selectors_group)
        self._dataset_combo.setFixedWidth(selection_combo_width)
        self._dataset_combo.addItems(['All', 'Recent', 'Eval', 'Train'])
        self._dataset_combo.setCurrentIndex(0)
        self._dataset_combo.setEnabled(False)
        self._dataset_combo.setToolTip('Dataset partitions used during train-eval process. Select "Recent" to include the updated matches.')
        selectors_layout.addWidget(dataset_label, 1, 0)
        selectors_layout.addWidget(self._dataset_combo, 1, 1)

        agent_label = QLabel('Agent ID', selectors_group)
        self._agent_combo = QComboBox(selectors_group)
        self._agent_combo.addItems(self._agent_database.get_agent_ids())
        self._agent_combo.setCurrentIndex(-1)
        self._agent_combo.setFixedWidth(selection_combo_width * 2 - 60)
        self._agent_combo.setToolTip('Unique agent identifier.')
        selectors_layout.addWidget(agent_label, 1, 3)
        selectors_layout.addWidget(self._agent_combo, 1, 4)

        selectors_layout.setColumnStretch(2, 1)

        page_layout.addWidget(selectors_group)

        filters_group = QGroupBox('Match Filters', page)
        filters_layout = QGridLayout(filters_group)
        filters_layout.setContentsMargins(12, 0, 12, 8)
        filters_layout.setHorizontalSpacing(12)
        filters_layout.setVerticalSpacing(8)

        filter_combo_width = 140
        odd_slider_width = 400
        percentile_slider_width = 150

        prediction_label = QLabel('Prediction', filters_group)
        self._prediction_combo = QComboBox(filters_group)
        self._prediction_combo.setFixedWidth(filter_combo_width)
        self._prediction_combo.setEnabled(False)
        filters_layout.addWidget(prediction_label, 0, 0)
        filters_layout.addWidget(self._prediction_combo, 0, 1)

        odd_label = QLabel('Odd', filters_group)
        self._odd_combo = QComboBox(filters_group)
        self._odd_combo.setFixedWidth(filter_combo_width)
        self._odd_combo.addItems(list(self._odd_limits.keys()))
        self._odd_combo.setEnabled(False)
        self._odd_combo.setToolTip('Selects the odd to apply the value range filter.')

        self._odd_slider = FloatDoubleRangeSlider(
            minimum=self._odd_limits['1'][0],
            maximum=self._odd_limits['1'][1],
            step=_ODD_FILTER_STEP,
            parent=filters_group
        )
        self._odd_slider.setFixedWidth(odd_slider_width)
        self._odd_slider.setEnabled(False)
        self._odd_slider.setToolTip('Filters the value range of the specified odd.')
        filters_layout.addWidget(odd_label, 1, 0)
        filters_layout.addWidget(self._odd_combo, 1, 1)
        filters_layout.addWidget(self._odd_slider, 1, 2, 1, 2)

        percentile_label = QLabel('Percentile', filters_group)
        self._percentile_combo = QComboBox(filters_group)
        self._percentile_combo.setFixedWidth(filter_combo_width)
        self._percentile_combo.setToolTip('Selects percentile of model\'s output.')
        percentile_slider_container = QWidget(filters_group)
        percentile_slider_container.setFixedWidth(percentile_slider_width)
        self._percentile_slider = QSlider(Qt.Orientation.Horizontal, percentile_slider_container)
        self._percentile_slider.setRange(0, 100)
        self._percentile_slider.setSingleStep(1)
        self._percentile_slider.setPageStep(1)
        self._percentile_slider.setValue(0)
        self._percentile_slider.setFixedWidth(percentile_slider_width)
        self._percentile_slider.setEnabled(False)
        percentile_slider_height = self._percentile_slider.sizeHint().height()
        self._percentile_slider.setGeometry(0, 0, percentile_slider_width, percentile_slider_height)
        self._percentile_slider.setToolTip('Filters the minimum predicted probability for the specified target variable.')
        self._percentile_value_label = QLabel('0', percentile_slider_container)
        self._percentile_value_label.adjustSize()
        percentile_slider_container.setFixedHeight(percentile_slider_height + self._percentile_value_label.height())
        self._probability_label = QLabel('Prob: 0.0%', filters_group)
        percentile_slider_layout = QHBoxLayout()
        percentile_slider_layout.setSpacing(6)
        percentile_slider_layout.addWidget(QLabel('0', filters_group))
        percentile_slider_layout.addWidget(percentile_slider_container)
        percentile_slider_layout.addWidget(QLabel('100', filters_group))
        percentile_slider_layout.addWidget(self._probability_label)
        percentile_slider_layout.addStretch()
        filters_layout.addWidget(percentile_label, 2, 0)
        filters_layout.addWidget(self._percentile_combo, 2, 1)
        filters_layout.addLayout(percentile_slider_layout, 2, 2, 1, 2)

        saved_filters_label = QLabel('Saved Filters', filters_group)
        self._saved_filters_combo = CheckableComboBox(filters_group)
        self._saved_filters_combo.setFixedWidth(self._target_combo.width())
        self._saved_filters_combo.setToolTip('Enable/Disable match filters.')
        filters_layout.addWidget(saved_filters_label, 3, 0)
        filters_layout.addWidget(self._saved_filters_combo, 3, 1)

        filter_buttons_layout = QHBoxLayout()
        filter_buttons_layout.setSpacing(6)
        self._reset_filter_button = QPushButton('Reset', filters_group)
        self._reset_filter_button.setFixedWidth(70)
        self._reset_filter_button.setEnabled(False)
        self._reset_filter_button.setToolTip('Reset current filter.')
        self._add_filter_button = QPushButton('Store', filters_group)
        self._add_filter_button.setFixedWidth(80)
        self._add_filter_button.setEnabled(False)
        self._add_filter_button.setToolTip('Store current filter into model filters.')
        self._remove_filter_button = QPushButton('Delete', filters_group)
        self._remove_filter_button.setFixedWidth(120)
        self._remove_filter_button.setEnabled(False)
        self._remove_filter_button.setToolTip('Deselects and remove all selected filters from the model filters.')
        self._clear_filters_button = QPushButton('Clear', filters_group)
        self._clear_filters_button.setFixedWidth(75)
        self._clear_filters_button.setEnabled(False)
        self._clear_filters_button.setToolTip('Deselects all selected filters, without removing them.')
        filter_buttons_layout.addWidget(self._reset_filter_button)
        filter_buttons_layout.addWidget(self._add_filter_button)
        filter_buttons_layout.addWidget(self._remove_filter_button)
        filter_buttons_layout.addWidget(self._clear_filters_button)
        filter_buttons_layout.addStretch()
        filters_layout.addLayout(filter_buttons_layout, 3, 2, 1, 2)

        filters_layout.setColumnStretch(3, 1)
        page_layout.addWidget(filters_group)

        metrics_group = QGroupBox('Evaluation Metrics', page)
        metrics_layout = QVBoxLayout(metrics_group)
        self._metrics_table = ExcelTable(
            parent=metrics_group,
            df=self._metrics_df,
            readonly=True,
            supports_sorting=False,
            supports_query_search=False,
            supports_deletion=False,
            auto_resize=True
        )
        self._metrics_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._metrics_table.verticalHeader().hide()
        metrics_table_height = (
            self._metrics_table.horizontalHeader().sizeHint().height()
            + self._metrics_table.verticalHeader().defaultSectionSize()
            + self._metrics_table.horizontalScrollBar().sizeHint().height()
            + self._metrics_table.frameWidth() * 2
            + 2
        )
        self._metrics_table.setFixedHeight(metrics_table_height)
        metrics_layout.addWidget(self._metrics_table)
        self._hide_extra_metrics_checkbox = QCheckBox('Hide Extra Metrics', metrics_group)
        self._hide_extra_metrics_checkbox.setChecked(False)
        self._hide_extra_metrics_checkbox.setToolTip('Hide extra evaluation metrics from the table.')
        self._double_chance_checkbox = QCheckBox('Switch to Double Chance', metrics_group)
        self._double_chance_checkbox.setChecked(False)
        self._double_chance_checkbox.setEnabled(False)
        self._double_chance_checkbox.setToolTip('Evaluate Result predictions as double-chance outcomes.')
        self._evaluate_yearly_button = QPushButton('Evaluate Yearly', metrics_group)
        self._evaluate_yearly_button.setToolTip('Show evaluation metrics grouped by season.')
        metrics_controls_layout = QHBoxLayout()
        metrics_controls_layout.setContentsMargins(0, 0, 0, 0)
        metrics_controls_layout.addWidget(self._hide_extra_metrics_checkbox)
        metrics_controls_layout.addWidget(self._double_chance_checkbox)
        metrics_controls_layout.addWidget(self._evaluate_yearly_button)
        metrics_controls_layout.addStretch()
        metrics_layout.addLayout(metrics_controls_layout)
        page_layout.addWidget(metrics_group)

        # Constructing dataframe for the table.
        table_df = self._df[[
            'Date', 'Season', 'Week', 'Home', 'Away', '1', 'X', '2', '1X', '12', 'X2', 'HG', 'AG', 'Result', 'Result-U/O'
        ]]
        table_df['Predicted'] = ''
        for column in self._probability_columns:
            table_df[column] = ''
        table_df['Betting'] = ''

        self._results_group = QGroupBox(f'Evaluation Results ({table_df.shape[0]} Matches)', page)
        results_layout = QVBoxLayout(self._results_group)
        self._matches_table = ExcelTable(
            parent=self._results_group,
            df=table_df,
            readonly=True,
            supports_sorting=False,
            supports_query_search=False,
            supports_deletion=False,
            auto_resize=False,
            allow_mouse_clicks=False
        )
        self._matches_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._matches_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        cols_to_hide = table_df.columns.tolist()[13:]
        self._matches_table.hide_columns(cols_to_hide, hide=True)
        results_table_height = (
            self._matches_table.horizontalHeader().sizeHint().height()
            + self._matches_table.verticalHeader().defaultSectionSize() * 10
            + self._matches_table.horizontalScrollBar().sizeHint().height()
            + self._matches_table.frameWidth() * 2
            + 2
        )
        self._matches_table.setMinimumHeight(results_table_height)
        results_layout.addWidget(self._matches_table)
        self._hide_match_columns_checkbox = QCheckBox('Hide Extra Columns', self._results_group)
        self._hide_match_columns_checkbox.setChecked(False)
        results_layout.addWidget(
            self._hide_match_columns_checkbox,
            alignment=Qt.AlignmentFlag.AlignLeft
        )
        self._hide_match_columns_checkbox.setToolTip('Hide Season, Week and Odd columns to simplify the table.')
        page_layout.addWidget(self._results_group, 1)

        scroll_area.setWidget(page)

        self._target_combo.currentIndexChanged.connect(self._target_changed)
        self._hide_extra_metrics_checkbox.toggled.connect(self._hide_extra_metrics_changed)
        self._double_chance_checkbox.toggled.connect(self._double_chance_changed)
        self._evaluate_yearly_button.clicked.connect(self._evaluate_yearly_clicked)
        self._hide_match_columns_checkbox.toggled.connect(self._hide_match_columns_changed)
        self._model_combo.currentIndexChanged.connect(self._model_changed)
        self._dataset_combo.currentIndexChanged.connect(self._dataset_changed)
        self._agent_combo.currentIndexChanged.connect(self._agent_changed)
        self._prediction_combo.currentIndexChanged.connect(self._prediction_changed)
        self._odd_combo.currentIndexChanged.connect(self._odd_changed)
        self._odd_slider.sliderReleased.connect(lambda: self._odd_slider_changed(*self._odd_slider.values()))
        self._percentile_combo.currentIndexChanged.connect(self._percentile_target_changed)
        self._percentile_slider.valueChanged.connect(self._update_percentile_slider_indicator)
        self._percentile_slider.sliderReleased.connect(self._percentile_slider_changed)
        self._saved_filters_combo.checkedItemsChanged.connect(self._saved_filter_changed)
        self._reset_filter_button.clicked.connect(self._reset_filter_clicked)
        self._add_filter_button.clicked.connect(self._add_filter_clicked)
        self._remove_filter_button.clicked.connect(self._remove_filter_clicked)
        self._clear_filters_button.clicked.connect(self._clear_filters_clicked)

    def _hide_extra_metrics_changed(self, checked: bool):
        self._metrics_table.hide_columns(columns=EXTRA_METRIC_COLUMNS, hide=checked)

    def _update_prediction_combo(self, double_chance: bool = False) -> None:
        if double_chance:
            prediction_labels = ['None', *DOUBLE_CHANCE_ODD_COLUMNS]
        else:
            prediction_labels = ['None', *self._target_type.get_mapper().keys()]

        with QSignalBlocker(self._prediction_combo):
            self._prediction_combo.clear()
            self._prediction_combo.addItems(prediction_labels)
            self._prediction_combo.setCurrentIndex(0)

    def _double_chance_changed(self, checked: bool):
        with QSignalBlocker(self._double_chance_checkbox):
            if checked and self._y_prob_dc is None:
                self._compute_double_chances()

            self._update_prediction_combo(double_chance=checked)

            probability_columns = [f'Prob-{result}' for result in TargetType.RESULT.get_mapper()]
            if checked:
                labels = {
                    column: f'Prob-{double_chance}'
                    for column, double_chance in zip(probability_columns, DOUBLE_CHANCE_ODD_COLUMNS)
                }
            else:
                labels = {}
            self._matches_table.set_column_labels(labels)

            metric_labels = {
                f'MIP-{odd}': f'MIP-{double_chance_odd}'
                for odd, double_chance_odd in zip(ODDS_COLUMNS, DOUBLE_CHANCE_ODD_COLUMNS)
            } if checked else {}
            self._metrics_table.set_column_labels(metric_labels)
            self._update_prediction_columns()
            self._reset_current_filter_controls()
            self._load_saved_filters()

        if self._y_pred_ids is None:
            self._update_metrics()
            self._highlight_matches()
        else:
            self._compile_eval_filters(eval_filters=self._get_eval_filters())

    def _evaluate_yearly_clicked(self, _checked: bool = False):
        if self._yearly_evaluation_dialog is not None and self._yearly_evaluation_dialog.isVisible():
            window_state = self._yearly_evaluation_dialog.windowState()
            if window_state & Qt.WindowState.WindowMinimized:
                self._yearly_evaluation_dialog.setWindowState(
                    (window_state & ~Qt.WindowState.WindowMinimized)
                    | Qt.WindowState.WindowActive
                )
            self._yearly_evaluation_dialog.raise_()
            self._yearly_evaluation_dialog.activateWindow()
            self._yearly_evaluation_dialog.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            return

        if self._yearly_evaluation_dialog is None:
            self._yearly_evaluation_dialog = YearlyEvaluationDialog(
                df=self._df,
                metric_columns=self._metrics_df.columns.tolist(),
                parent=self
            )

        self._yearly_evaluation_dialog.show()
        self._update_yearly_evaluation_dialog()
        self._yearly_evaluation_dialog.raise_()
        self._yearly_evaluation_dialog.activateWindow()

    def _update_yearly_evaluation_dialog(self):
        if self._yearly_evaluation_dialog is None or not self._yearly_evaluation_dialog.isVisible():
            return

        double_chance = self._double_chance_checkbox.isChecked()
        correct_ids = self._correct_dc_ids if double_chance else self._correct_ids
        has_model = correct_ids is not None
        self._yearly_evaluation_dialog.update_metrics(
            selected_ids=self._visible_match_ids(),
            current_metrics_df=self._metrics_df,
            target_type=self._target_type if has_model else None,
            y_pred_ids=(self._y_pred_dc if double_chance else self._y_pred_ids) if has_model else None,
            y_prob=(self._y_prob_dc if double_chance else self._y_prob) if has_model else None,
            actions=self._actions,
            double_chance=double_chance
        )

    def _hide_match_columns_changed(self, checked: bool):
        self._matches_table.hide_columns(columns=['Season', 'Week'] + ODDS_COLUMNS + DOUBLE_CHANCE_ODD_COLUMNS, hide=checked)

    def _visible_match_ids(self) -> pd.Index:
        """Return row IDs allowed by both the dataset and active filters."""

        selected_ids = self._df.index if self._dataset_ids is None else self._dataset_ids
        if self._odd_percentile_filter_ids is not None:
            selected_ids = selected_ids.intersection(self._odd_percentile_filter_ids, sort=False)
        return selected_ids

    def _update_match_view(self) -> None:
        """Apply row visibility before restoring the correct-row highlights."""

        visible_ids = self._visible_match_ids()
        hidden_ids = self._df.index.difference(visible_ids)
        self._matches_table.hide_rows(row_ids=hidden_ids.tolist())
        self._results_group.setTitle(f'Evaluation Results ({visible_ids.shape[0]} Matches)')
        self._highlight_matches()

    def _highlight_matches(self):
        correct_ids = self._correct_dc_ids if self._double_chance_checkbox.isChecked() else self._correct_ids
        if correct_ids is None:
            self._matches_table.clearSelection()
            return

        highlighted_ids = self._visible_match_ids().intersection(correct_ids, sort=False)
        self._matches_table.highlight_rows(row_ids=highlighted_ids.tolist())

    def _update_metrics(self):
        selected_ids = self._visible_match_ids()
        model_metric_columns = CLASSIFICATION_METRICS + PROBABILITY_METRICS + SKILL_METRICS
        double_chance = self._double_chance_checkbox.isChecked()
        correct_ids = self._correct_dc_ids if double_chance else self._correct_ids

        # --- Updating Model Metrics ---
        # Case 1. No model is selected:
        if correct_ids is None:
            self._metrics_df.loc[0, model_metric_columns] = np.nan
        else:
            # Case 2. No matches are selected.
            if selected_ids.shape[0] == 0:
                self._metrics_df.loc[0, model_metric_columns] = np.nan
            else:
                # Case 3. Evaluate model on selected matches.
                if double_chance:
                    metrics_df = generate_report_dc(
                        y_true=self._df.loc[selected_ids, TargetType.RESULT.value].to_numpy(),
                        y_pred=self._y_pred_dc[selected_ids],
                        y_prob=self._y_prob_dc[selected_ids],
                        odds=self._df.loc[selected_ids, DOUBLE_CHANCE_ODD_COLUMNS]
                    )
                    metrics_df.rename(columns={
                        f'MIP-{double_chance_odd}': f'MIP-{odd}'
                        for odd, double_chance_odd in zip(ODDS_COLUMNS, DOUBLE_CHANCE_ODD_COLUMNS)
                    }, inplace=True)
                else:
                    mapper = self._target_type.get_mapper()
                    y_true_labels = self._df.loc[selected_ids, self._target_type.value]
                    metrics_df = generate_report(
                        y_true=y_true_labels.map(mapper).to_numpy(),
                        y_pred=self._y_pred_ids[selected_ids],
                        y_prob=self._y_prob[selected_ids],
                        odds=self._df.loc[selected_ids, ODDS_COLUMNS],
                        target_type=self._target_type
                    )
                self._metrics_df.loc[0, model_metric_columns] = np.nan
                self._metrics_df.update(metrics_df)

        # --- Updating Agent Metrics --- (if an agent has been selected)
        if self._actions is not None:
            # Case 1: No matches are selected.
            if selected_ids.shape[0] == 0:
                self._metrics_df.loc[0, BETTING_PERFORMANCE_METRICS] = np.nan
            else:
                # Case 2: All has been selected.
                if selected_ids.shape[0] == self._df.shape[0]:
                    self._metrics_df.loc[0, BETTING_PERFORMANCE_METRICS] = self._total_betting_metrics
                else:
                    # Case 3: Evaluate agent on selected matches.
                    betting_metrics = compute_betting_metrics(df=self._df.loc[selected_ids], actions=self._actions[selected_ids])
                    self._metrics_df.loc[0, BETTING_PERFORMANCE_METRICS] = list(betting_metrics)

        # Updating metrics table
        self._metrics_table.modify_rows(row_ids=[0], items=self._metrics_df.values.tolist())
        self._update_yearly_evaluation_dialog()

    def _compile_eval_filters(self, eval_filters: List[EvalFilter]):
        # Compile filters against the complete dataset. The selected partition is
        # intersected separately so changing partitions cannot invalidate a filter.
        has_active_filter = any(
            eval_filter.prediction_filter is not None
            or eval_filter.odd_filter is not None
            or eval_filter.percentile_filter is not None
            for eval_filter in eval_filters
        )

        if not has_active_filter:
            self._odd_percentile_filter_ids = self._df.index
        else:
            # Filtering needs only odds, predictions, and probability outputs.
            # Copying the complete feature dataframe here is costly for large leagues.
            df = self._df[ODDS_COLUMNS + DOUBLE_CHANCE_ODD_COLUMNS].copy()
            probability_columns = [f'Prob-{col}' for col in self._target_type.get_mapper()]
            df[probability_columns] = self._y_prob

            if self._double_chance_checkbox.isChecked():
                double_chance_columns = [f'Prob-{col}' for col in DOUBLE_CHANCE_ODD_COLUMNS]
                df[double_chance_columns] = self._y_prob_dc
                df['Prediction'] = np.argmax(self._y_prob_dc, axis=1)
            else:
                df['Prediction'] = self._y_pred_ids

            filtered_df = filter_matches(matches_df=df, eval_filters=eval_filters)
            self._odd_percentile_filter_ids = filtered_df.index

        # Hiding rows invalidates the table selection, so restore highlights
        # only after applying the combined dataset/filter visibility.
        self._update_match_view()
        self._update_metrics()

    def _compute_double_chances(self):
        if self._target_type != TargetType.RESULT or self._y_prob is None:
            return

        self._y_prob_dc = np.column_stack((
            self._y_prob[:, 0] + self._y_prob[:, 1],
            self._y_prob[:, 0] + self._y_prob[:, 2],
            self._y_prob[:, 1] + self._y_prob[:, 2]
        ))
        y_pred_dc_ids = np.argmax(self._y_prob_dc, axis=1)
        self._y_pred_dc = np.asarray(DOUBLE_CHANCE_ODD_COLUMNS)[y_pred_dc_ids]

        results = self._df[TargetType.RESULT.value].to_numpy()
        correct_mask = (
            ((self._y_pred_dc == '1X') & np.isin(results, ['H', 'D']))
            | ((self._y_pred_dc == '12') & np.isin(results, ['H', 'A']))
            | ((self._y_pred_dc == 'X2') & np.isin(results, ['D', 'A']))
        )
        self._correct_dc_ids = self._df.index[correct_mask]

    def _update_prediction_columns(self):
        if self._y_pred is None or self._y_prob is None or self._target_type is None:
            return

        double_chance = self._double_chance_checkbox.isChecked()
        if double_chance:
            if self._y_pred_dc is None or self._y_prob_dc is None:
                return
            y_pred = self._y_pred_dc
            y_prob = self._y_prob_dc
        else:
            y_pred = self._y_pred
            y_prob = self._y_prob

        prediction_columns = ['Predicted', *[f'Prob-{label}' for label in self._target_type.get_mapper()]]
        prediction_data = np.column_stack((y_pred, y_prob.round(decimals=3))).tolist()
        self._matches_table.modify_columns(
            columns=prediction_columns,
            data=prediction_data,
            rows=list(range(self._df.shape[0]))
        )

    def _target_changed(self, _index: int):
        # Computing previous columns to hide.
        cols_to_hide = [] if self._target_type is None else self._target_columns[self._target_type]

        # Double chance is available only for Result models.
        with QSignalBlocker(self._double_chance_checkbox):
            self._double_chance_checkbox.setChecked(False)
        if self._double_chance_checkbox.isChecked():
            self._update_prediction_columns()
        self._matches_table.set_column_labels({})
        self._metrics_table.set_column_labels({})

        # Updating target type.
        self._target_type = self._target_combo.currentData()
        self._double_chance_checkbox.setEnabled(self._target_type == TargetType.RESULT)
        self._update_prediction_combo()

        # Adding model ids to combobox.
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItems(self._model_ids[self._target_type])
        self._model_combo.blockSignals(False)

        # De-selecting model and triggering model change callback.
        self._model_combo.setCurrentIndex(-1)

        # Clearing old percentile filters.
        self._percentile_combo.blockSignals(True)
        self._percentile_combo.clear()
        self._percentile_combo.addItems(list(self._target_type.get_mapper().keys()))
        self._percentile_combo.blockSignals(False)

        # Resetting table target columns.
        if cols_to_hide:
            self._matches_table.hide_columns(columns=cols_to_hide, hide=True)
        self._matches_table.hide_columns(columns=[self._target_type.value], hide=False)

    def _model_changed(self, _index: int):
        # Fetching selected model id.
        model_id = self._model_combo.currentText()
        self._odd_percentile_filter_ids = None
        self._y_prob_dc = None
        self._y_pred_dc = None
        self._correct_dc_ids = None

        with QSignalBlocker(self._prediction_combo):
            self._prediction_combo.setCurrentIndex(0)

        self._percentile_slider.blockSignals(True)
        self._percentile_slider.setValue(0)
        self._percentile_slider.blockSignals(False)
        self._min_probability = 0.0

        odd = self._odd_combo.currentText()
        minimum, maximum = self._odd_limits[odd]
        self._odd_slider.blockSignals(True)
        self._odd_slider.setRange(minimum=minimum, maximum=maximum)
        self._odd_slider.setValues(lower=minimum, upper=maximum)
        self._odd_slider.blockSignals(False)

        self._saved_filters_combo.blockSignals(True)
        self._saved_filters_combo.clear()

        if model_id == '':
            # Resetting table and metrics.
            prediction_columns = [
                'Predicted',
                *[f'Prob-{label}' for label in self._target_type.get_mapper()]
            ]
            empty_predictions = [[''] * len(prediction_columns) for _ in range(self._df.shape[0])]
            self._matches_table.modify_columns(
                columns=prediction_columns,
                data=empty_predictions,
                rows=list(range(self._df.shape[0]))
            )
            self._matches_table.hide_columns(columns=prediction_columns, hide=True)
            self._y_pred_ids = None
            self._y_pred = None
            self._y_prob = None
            self._correct_ids = None
            self._odd_percentile_filter_ids = None

            # Resetting dataset partition to "All" and force-triggering dataset change callback.
            with QSignalBlocker(self._dataset_combo):
                self._dataset_combo.setCurrentIndex(0)
            self._dataset_changed(0)

            # Disabling evaluation and dataset combo.
            self._dataset_combo.setEnabled(False)
            self._prediction_combo.setEnabled(False)
            self._odd_combo.setEnabled(False)
            self._odd_slider.setEnabled(False)
            self._percentile_combo.setEnabled(False)
            self._percentile_slider.setEnabled(False)
            self._reset_filter_button.setEnabled(False)
            self._add_filter_button.setEnabled(False)
            self._remove_filter_button.setEnabled(False)
            self._clear_filters_button.setEnabled(False)
            self._saved_filters_combo.blockSignals(False)
            return

        target_type = self._target_combo.currentData()

        # Loading model.
        model, self._model_config = self._model_database.load_model(model_id=model_id)

        # Computing dataset partitions.
        num_recent_samples = self._df.shape[0] - self._model_config.num_train_samples - self._model_config.num_eval_samples
        index = self._df.index
        self._recent_ids = index[:num_recent_samples]
        self._eval_ids = index[num_recent_samples: num_recent_samples + self._model_config.num_eval_samples]
        self._train_ids = index[-self._model_config.num_train_samples:]

        # Adding evaluation filters for the selected prediction mode.
        self._load_saved_filters()

        # Enabling evaluation widgets.
        self._dataset_combo.setEnabled(True)
        self._prediction_combo.setEnabled(True)
        self._odd_combo.setEnabled(True)
        self._odd_slider.setEnabled(True)
        self._percentile_combo.setEnabled(True)
        self._percentile_slider.setEnabled(True)
        self._reset_filter_button.setEnabled(True)
        self._add_filter_button.setEnabled(True)
        self._remove_filter_button.setEnabled(True)
        self._clear_filters_button.setEnabled(True)
        self._saved_filters_combo.blockSignals(False)

        # Generating predictions.
        mapper = target_type.get_mapper()
        label_names = np.array(list(mapper.keys()))

        self._y_prob = model.predict_proba(df=self._df)
        self._y_pred_ids = np.argmax(self._y_prob, axis=1)
        self._y_pred = label_names[self._y_pred_ids]
        correct_mask = self._df[target_type.value].to_numpy() == self._y_pred
        self._correct_ids = self._df.index[correct_mask]

        if self._double_chance_checkbox.isChecked():
            self._compute_double_chances()

        # Updating table and metrics.
        # Displaying table result/predicted columns.
        self._update_prediction_columns()
        self._matches_table.hide_columns(columns=['Predicted'] + self._target_columns[self._target_type], hide=False)

        # Resetting dataset partition to "All" and force-triggering dataset change callback.
        with QSignalBlocker(self._dataset_combo):
            self._dataset_combo.setCurrentIndex(0)
        self._dataset_changed(0)

    def _dataset_changed(self, _index: int):
        partition = self._dataset_combo.currentText()

        if partition == 'All':
            self._dataset_ids = None
        else:
            if partition == 'Recent':
                self._dataset_ids = self._recent_ids
            elif partition == 'Eval':
                self._dataset_ids = self._eval_ids
            elif partition == 'Train':
                self._dataset_ids = self._train_ids
            else:
                raise NotImplementedError(f'Partition: "{partition}" has not been implemented yet.')

        # Updating table and metrics.
        self._update_match_view()
        self._update_metrics()

    def _agent_changed(self, _index: int):
        # Fetching selected model id.
        agent_id = self._agent_combo.currentText()

        # Loading the agent and generating actions in a worker process.
        task = Task(
            _load_agent_and_compute_actions,
            title='Agent Evaluation',
            description=f'Evaluating agent "{agent_id}". Please wait...',
            parent=self,
            agent_id=agent_id,
            agent_database=self._agent_database,
            df=self._df
        )
        result = task.run()
        if result is None:
            return

        self._actions, _, total_betting_metrics = result
        self._total_betting_metrics = list(total_betting_metrics.values())
        actions = [Action.get_name(action=action) for action in self._actions]

        # Updating metrics.
        self._metrics_df.loc[0, BETTING_PERFORMANCE_METRICS] = self._total_betting_metrics
        self._metrics_table.modify_rows(row_ids=[0], items=self._metrics_df.values.tolist())
        self._update_yearly_evaluation_dialog()

        # Updating table.
        self._matches_table.hide_columns(columns=['Betting'], hide=False)
        action_data = [[action] for action in actions]
        self._matches_table.modify_columns(columns=['Betting'], data=action_data, rows=list(range(self._df.shape[0])))

    def _compile_current_filter(self) -> EvalFilter:
        prediction = self._prediction_combo.currentText()
        if prediction == 'None':
            prediction_filter = None
        elif self._double_chance_checkbox.isChecked():
            prediction_filter = DOUBLE_CHANCE_ODD_COLUMNS.index(prediction)
        else:
            prediction_filter = self._target_type.get_mapper()[prediction]

        odd = self._odd_combo.currentText()
        minimum, maximum = self._odd_slider.values()

        if minimum == self._odd_slider.minimum and maximum == self._odd_slider.maximum:
            odd_filter = None
        else:
            odd_filter = (odd, minimum, maximum)

        # Compiling percentile filter.
        if self._min_probability == 0.0:
            percentile_filter = None
        else:
            result_index = self._percentile_combo.currentIndex()
            result = list(self._target_type.get_mapper().keys())[result_index]
            percentile_filter = (result, self._min_probability, self._percentile_slider.value())

        return EvalFilter(
            prediction_filter=prediction_filter,
            odd_filter=odd_filter,
            percentile_filter=percentile_filter,
            double_chance=self._double_chance_checkbox.isChecked(),
        )

    def _load_saved_filters(self) -> None:
        """Show only saved filters created for the active prediction mode."""

        model_selected = bool(self._model_combo.currentText())
        saved_filters = (
            list(self._model_config.eval_filters or [])
            if model_selected and self._model_config is not None
            else []
        )
        double_chance = self._double_chance_checkbox.isChecked()

        with QSignalBlocker(self._saved_filters_combo):
            self._saved_filters_combo.clear()
            for eval_filter in saved_filters:
                if eval_filter.double_chance != double_chance:
                    continue
                self._saved_filters_combo.addItem(
                    self._format_eval_filter(eval_filter),
                    userData=eval_filter,
                )

    def _format_eval_filter(self, eval_filter: EvalFilter) -> str:
        if eval_filter.prediction_filter is None:
            prediction_text = 'None'
        else:
            has_none_option = self._prediction_combo.itemText(0) == 'None'
            combo_index = eval_filter.prediction_filter + int(has_none_option)
            prediction_text = self._prediction_combo.itemText(combo_index)

        if eval_filter.odd_filter is None:
            odd_text = 'None'
        else:
            odd, minimum, maximum = eval_filter.odd_filter
            odd_text = f'{odd}:({round(minimum, 1)}-{round(maximum, 1)})'

        if eval_filter.percentile_filter is None:
            percentile_text = 'None'
        else:
            target, min_probability, percentile = eval_filter.percentile_filter
            percentile_text = f'{target}:({round(min_probability, 2)},{percentile})'

        return f'{prediction_text} | {odd_text} | {percentile_text}'

    def _selected_saved_filter_indexes(self) -> List[int]:
        model = self._saved_filters_combo.model()
        return [
            index
            for index in range(self._saved_filters_combo.count())
            if model.item(index).checkState() == Qt.CheckState.Checked
        ]

    def _get_eval_filters(self) -> List[EvalFilter]:
        # Compiling current filter.
        eval_filters = [self._compile_current_filter()]

        # Appending the selected saved filters stored in the combobox items.
        for index in self._selected_saved_filter_indexes():
            eval_filters.append(self._saved_filters_combo.itemData(index))

        return eval_filters

    def _prediction_changed(self, _index: int):
        eval_filters = self._get_eval_filters()
        self._compile_eval_filters(eval_filters=eval_filters)

    def _odd_changed(self, _index: int):
        odd = self._odd_combo.currentText()
        minimum, maximum = self._odd_limits[odd]

        self._odd_slider.blockSignals(True)
        self._odd_slider.setRange(minimum=minimum, maximum=maximum)
        self._odd_slider.setValues(lower=minimum, upper=maximum)
        self._odd_slider.blockSignals(False)

        eval_filters = self._get_eval_filters()
        self._compile_eval_filters(eval_filters=eval_filters)

    def _odd_slider_changed(self, _minimum: float, _maximum: float):
        eval_filters = self._get_eval_filters()
        self._compile_eval_filters(eval_filters=eval_filters)

    def _percentile_target_changed(self, _index: int):
        self._percentile_slider.setValue(0)

    def _percentile_slider_changed(self):
        eval_filters = self._get_eval_filters()
        self._compile_eval_filters(eval_filters=eval_filters)

    def _update_percentile_slider_indicator(self, value: int):
        percentile = self._percentile_slider.value()

        if percentile == 0:
            self._min_probability = 0.0
        else:
            result_index = self._percentile_combo.currentIndex()
            self._min_probability = np.percentile(self._y_prob[:, result_index], percentile)

        self._probability_label.setText(f'Prob: {round(self._min_probability * 100, 1)}%')
        self._percentile_value_label.setText(str(value))
        self._percentile_value_label.adjustSize()

        handle_width = self._percentile_slider.style().pixelMetric(QStyle.PixelMetric.PM_SliderLength)
        available_width = self._percentile_slider.width() - handle_width
        handle_position = QStyle.sliderPositionFromValue(self._percentile_slider.minimum(), self._percentile_slider.maximum(), value, available_width)
        label_x = handle_position + handle_width // 2 - self._percentile_value_label.width() // 2
        label_x = max(0, min(label_x, self._percentile_slider.width() - self._percentile_value_label.width()))
        self._percentile_value_label.move(label_x, self._percentile_slider.height())

    def _saved_filter_changed(self):
        eval_filters = self._get_eval_filters()
        self._compile_eval_filters(eval_filters=eval_filters)

    def _reset_filter_clicked(self, _checked: bool = False):
        with (
                QSignalBlocker(self._prediction_combo),
                QSignalBlocker(self._odd_slider),
                QSignalBlocker(self._percentile_slider)
        ):
            self._prediction_combo.setCurrentIndex(0)
            odd = self._odd_combo.currentText()
            minimum, maximum = self._odd_limits[odd]
            self._odd_slider.setRange(minimum=minimum, maximum=maximum)
            self._odd_slider.setValues(lower=minimum, upper=maximum)
            self._percentile_slider.setValue(0)

        self._min_probability = 0.0
        self._update_percentile_slider_indicator(value=0)
        self._compile_eval_filters(eval_filters=self._get_eval_filters())

    def _reset_current_filter_controls(self) -> None:
        """Reset the filter editor without applying the reset values."""

        with (
                QSignalBlocker(self._prediction_combo),
                QSignalBlocker(self._odd_combo),
                QSignalBlocker(self._odd_slider),
                QSignalBlocker(self._percentile_combo),
                QSignalBlocker(self._percentile_slider)
        ):
            self._prediction_combo.setCurrentIndex(0)
            self._odd_combo.setCurrentIndex(0)
            odd = self._odd_combo.currentText()
            minimum, maximum = self._odd_limits[odd]
            self._odd_slider.setRange(minimum=minimum, maximum=maximum)
            self._odd_slider.setValues(lower=minimum, upper=maximum)

            self._percentile_combo.setCurrentIndex(0)
            self._percentile_slider.setValue(0)

        self._min_probability = 0.0
        self._update_percentile_slider_indicator(value=0)

    def _store_model_config_async(self) -> None:
        """Persist the current model config without blocking the GUI thread."""

        _MODEL_CONFIG_WRITER.submit(self._model_database.update_model_config, model_config=self._model_config)

    def _add_filter_clicked(self, _checked: bool = False):
        eval_filter = self._compile_current_filter()

        # Add the filter to a new copy of the immutable model config.
        saved_filters = list(self._model_config.eval_filters or [])
        if eval_filter in saved_filters:
            return

        saved_filters.append(eval_filter)
        self._model_config = replace(self._model_config, eval_filters=saved_filters)

        # Persist the updated model config.
        self._store_model_config_async()

        # Add and select the saved filter, then apply the resulting filter list once.
        index = self._saved_filters_combo.count()
        self._saved_filters_combo.addItem(self._format_eval_filter(eval_filter), userData=eval_filter)
        self._saved_filters_combo.model().item(index).setCheckState(Qt.CheckState.Checked)
        self._saved_filter_changed()
        self._reset_current_filter_controls()

    def _remove_filter_clicked(self, _checked: bool = False):
        selected_indexes = self._selected_saved_filter_indexes()

        if not selected_indexes:
            return

        # Remove one stored occurrence for every selected combobox item.
        selected_filters = [self._saved_filters_combo.itemData(index) for index in selected_indexes]
        saved_filters = list(self._model_config.eval_filters or [])
        for eval_filter in selected_filters:
            if eval_filter in saved_filters:
                saved_filters.remove(eval_filter)

        self._model_config = replace(self._model_config, eval_filters=saved_filters)
        self._store_model_config_async()

        # Remove items from highest to lowest index, then apply the remaining filters once.
        for index in reversed(selected_indexes):
            self._saved_filters_combo.removeItem(index)
        self._saved_filter_changed()

    def _clear_filters_clicked(self, _checked: bool = False):
        selected_indexes = self._selected_saved_filter_indexes()

        if not selected_indexes:
            return

        # Deselect every saved filter, then apply the current filter once.
        model = self._saved_filters_combo.model()
        for index in selected_indexes:
            model.item(index).setCheckState(Qt.CheckState.Unchecked)
        self._saved_filter_changed()
