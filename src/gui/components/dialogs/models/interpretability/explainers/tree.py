from typing import List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QValueAxis
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QCursor, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QTabBar,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from src.core.database.model import ModelDatabase
from src.core.interpretability.explainers import DecisionTreeExplainer
from src.gui.animations.barplot import BarplotScaling
from src.gui.components.dialogs.analysis.reports.rules import RulesPlot
from src.gui.components.dialogs.models.interpretability.explainers.base import ExplainerBaseDialog, _increase_font
from src.gui.widgets.table import ExcelTable


class _FeatureImportanceChart(QChartView):
    """Bar chart of the strongest decision-tree feature importances."""

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

    def show_message(self, message: str):
        self._reset()
        self._chart.setTitle(message)

    def set_importances(self, importances: Optional[pd.DataFrame], ascending: bool = False):
        self._reset()

        if importances is None or not {'Feature', 'Importance'}.issubset(importances.columns):
            self.show_message('No feature-importance results are available.')
            return

        values = pd.to_numeric(importances['Importance'], errors='coerce')
        valid = values.notna() & np.isfinite(values)
        if not valid.any():
            self.show_message('No feature-importance results are available.')
            return

        feature_names = importances.loc[valid, 'Feature'].astype(str).tolist()
        importance_values = values.loc[valid].to_numpy(dtype=float)
        if ascending:
            order = np.argsort(importance_values, kind='stable').tolist()[:15]
        else:
            order = np.argsort(-importance_values, kind='stable').tolist()[:15]

        self._feature_names = [feature_names[index] for index in order]
        self._hover_values = importance_values[order]

        bar_set = QBarSet('Importance')
        bar_set.append(self._hover_values.tolist())
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
        axis_maximum = max(float(self._hover_values.max()), 0.0)
        value_axis.setRange(0.0, max(axis_maximum * 1.1, 0.01))
        value_axis.applyNiceNumbers()

        self._chart.addSeries(series)
        self._chart.addAxis(feature_axis, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(value_axis, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(feature_axis)
        series.attachAxis(value_axis)

        title_font = _increase_font(feature_title_font)
        title_font.setBold(True)
        self._chart.setTitleFont(title_font)
        self._chart.setTitle(f'Feature Importance Analysis of {len(self._feature_names)} Best Features')
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignTop)

        self._barplot_scalings = [
            BarplotScaling(bar_set, self._hover_values.tolist(), parent=self)
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


class DecisionTreeExplainerDialog(ExplainerBaseDialog):
    """Dialog for interpreting a stored decision-tree model."""

    MAX_RENDER_DEPTH = 6
    INITIAL_RENDER_DELAY_MS = 100

    def __init__(
            self,
            df: pd.DataFrame,
            model_id: str,
            model_database: ModelDatabase,
            parent: Optional[QWidget] = None,
    ):
        self._tree_model = None
        self._feature_importances = None

        self._rule_tabs = None
        self._rule_tree_index = -1
        self._rule_leaf_index = -1
        self._rule_plot = None
        self._rule_tree_render_attempted = False
        self._rule_tree_rendered = False
        self._rule_tree_error = ''
        self._rule_leaf_rendered = False
        self._tree_render_in_progress = False

        self._outcome_split_combo = None
        self._outcome_match_spinner = None
        self._outcome_table = None
        self._outcome_plot = None
        self._outcome_rendered_key = None

        self._importance_sort_combo = None
        self._importance_chart = None

        self._initial_tree_views_rendered = False
        self._initial_render_timer = None

        explainer = DecisionTreeExplainer(df=df, model_id=model_id, database=model_database)
        self._configured_max_depth = explainer.model_config.classifier_config.get('max_depth')
        self._fitted_tree_depth = explainer.get_tree_depth()
        self._tree_rendering_blocked = self._depth_exceeds_render_limit(
            configured_depth=self._configured_max_depth,
            fitted_depth=self._fitted_tree_depth,
        )

        super().__init__(explainer=explainer, parent=parent)
        self._initial_render_timer = QTimer(self)
        self._initial_render_timer.setSingleShot(True)
        self._initial_render_timer.timeout.connect(self._render_initial_tree_views)
        self.setWindowTitle(model_id)

    @classmethod
    def _depth_exceeds_render_limit(cls, configured_depth, fitted_depth) -> bool:
        """Guard configured and unbounded trees before invoking dtreeviz."""

        def normalize_depth(depth):
            if depth is None:
                return None
            try:
                return int(depth)
            except (TypeError, ValueError, OverflowError):
                return None

        configured_depth = normalize_depth(configured_depth)
        fitted_depth = normalize_depth(fitted_depth)
        configured_too_deep = (
            configured_depth is not None
            and configured_depth > cls.MAX_RENDER_DEPTH
        )
        fitted_too_deep = (
            fitted_depth is not None
            and fitted_depth > cls.MAX_RENDER_DEPTH
        )
        configured_depth_is_unbounded = configured_depth is None or configured_depth <= 0
        depth_is_unknown = configured_depth_is_unbounded and fitted_depth is None
        return configured_too_deep or fitted_too_deep or depth_is_unknown

    def explainer_model(self):
        """Compute SHAP data and prepare non-Qt tree data in the worker task."""

        super().explainer_model()
        self._feature_importances = self._explainer.get_feature_importances()

        if not self._tree_rendering_blocked:
            self._tree_model = self._explainer.get_tree_model()

    def build_analysis_frames(self):
        self._build_feature_importance_frame()

        if self._tree_rendering_blocked:
            self._build_depth_warning()
            return

        self._build_rule_extraction_frame()
        self._build_outcome_explanation_frame()

    def showEvent(self, event):
        super().showEvent(event)

        if (
                self._tree_rendering_blocked
                or self._initial_tree_views_rendered
                or self._initial_render_timer.isActive()
        ):
            return

        self._initial_render_timer.start(self.INITIAL_RENDER_DELAY_MS)

    def _tree_depth_details(self) -> str:
        configured_depth = (
            'unlimited'
            if self._configured_max_depth is None
            else str(self._configured_max_depth)
        )
        fitted_depth = (
            'unknown'
            if self._fitted_tree_depth is None
            else str(self._fitted_tree_depth)
        )
        return f'Configured depth: {configured_depth}; fitted depth: {fitted_depth}.'

    def _build_depth_warning(self):
        warning = QLabel(
            (
                'Decision-tree and outcome rendering are unavailable. '
                f'Visualizations are limited to a maximum tree depth of {self.MAX_RENDER_DEPTH}. '
                f'{self._tree_depth_details()}'
            ),
            self,
        )
        warning.setObjectName('treeDepthWarning')
        warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        warning.setWordWrap(True)
        warning.setFixedWidth(self.ANALYSIS_FRAME_WIDTH)
        warning.setMinimumHeight(72)
        warning_font = _increase_font(warning.font())
        warning_font.setBold(True)
        warning.setFont(warning_font)
        self._analysis_layout.addWidget(warning)

    def _build_rule_extraction_frame(self):
        rule_frame = QGroupBox('Rule Extraction', self)
        self._set_analysis_frame_size(rule_frame)
        rule_layout = QVBoxLayout(rule_frame)
        rule_layout.setContentsMargins(12, 12, 12, 12)
        rule_layout.setSpacing(8)

        tabs_layout = QHBoxLayout()
        self._rule_tabs = QTabBar(rule_frame)
        self._rule_tabs.setExpanding(False)
        self._rule_tabs.setDrawBase(False)
        self._rule_tree_index = self._rule_tabs.addTab('Tree Model')
        self._rule_leaf_index = self._rule_tabs.addTab('Leaf Distributions')

        tab_spacing = QWidget(self._rule_tabs)
        tab_spacing.setFixedWidth(12)
        tab_spacing.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._rule_tabs.setTabButton(
            self._rule_tree_index,
            QTabBar.ButtonPosition.RightSide,
            tab_spacing,
        )

        tabs_layout.addWidget(self._rule_tabs)
        tabs_layout.addStretch(1)
        rule_layout.addLayout(tabs_layout)

        self._rule_plot = RulesPlot(rule_frame)
        rule_layout.addWidget(self._rule_plot, 1)
        self._analysis_layout.addWidget(rule_frame, 1)

        self._rule_tabs.currentChanged.connect(self._rule_tab_changed)
        self._rule_plot.show_message('The decision-tree visualization is waiting to render.')

    def _render_rule_tree_once(self):
        if self._rule_tree_render_attempted or self._tree_render_in_progress:
            return

        self._rule_tree_render_attempted = True
        if self._tree_model is None:
            self._rule_tree_error = 'The decision-tree model could not be prepared for visualization.'
            self._rule_plot.show_message(self._rule_tree_error)
            return

        try:
            self._tree_render_in_progress = True
            rendered = self._rule_plot.show_tree(
                self._tree_model,
                show_progress=True,
            )
            if rendered:
                self._rule_tree_rendered = True
            else:
                self._rule_tree_error = 'Decision-tree rendering was cancelled or failed.'
                self._rule_plot.show_message(self._rule_tree_error)
        except Exception as error:
            self._rule_tree_error = f'Unable to render the decision tree.\n{error}'
            self._rule_plot.show_message(self._rule_tree_error)
        finally:
            self._tree_render_in_progress = False

    def _rule_tab_changed(self, index: int):
        if index == self._rule_tree_index:
            if self._rule_tree_rendered:
                self._rule_plot.show_tree_page()
            elif self._rule_tree_error:
                self._rule_plot.show_message(self._rule_tree_error)
            else:
                self._render_rule_tree_once()
            return

        if self._tree_model is None:
            self._rule_plot.show_message(
                'The decision-tree model could not be prepared for visualization.'
            )
            return

        if self._rule_leaf_rendered:
            self._rule_plot.show_matplotlib_page()
            return

        self._rule_plot.show_leaf_distributions(self._tree_model)
        self._rule_leaf_rendered = (
            self._rule_plot.currentWidget() is self._rule_plot._canvas
        )

    def _build_outcome_explanation_frame(self):
        outcome_frame = QGroupBox('Outcome Explanation', self)
        self._set_analysis_frame_size(outcome_frame)
        outcome_layout = QVBoxLayout(outcome_frame)
        outcome_layout.setContentsMargins(12, 12, 12, 12)
        outcome_layout.setSpacing(8)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel('Dataset', outcome_frame))
        self._outcome_split_combo = QComboBox(outcome_frame)
        self._outcome_split_combo.addItems(['Train', 'Test'])
        controls_layout.addWidget(self._outcome_split_combo)
        controls_layout.addSpacing(16)
        controls_layout.addWidget(QLabel('Match ID', outcome_frame))

        self._outcome_match_spinner = QSpinBox(outcome_frame)
        self._outcome_match_spinner.setFixedWidth(100)
        self._outcome_match_spinner.setRange(
            0,
            max(len(self._explainer.train_df) - 1, 0),
        )
        self._outcome_match_spinner.setValue(0)
        self._outcome_match_spinner.setEnabled(not self._explainer.train_df.empty)
        controls_layout.addWidget(self._outcome_match_spinner)
        controls_layout.addStretch(1)
        outcome_layout.addLayout(controls_layout)

        initial_row = self._explainer.train_df.head(1).reset_index(drop=True)
        if initial_row.empty:
            initial_row = pd.DataFrame(
                [[None] * len(self._explainer.train_df.columns)],
                columns=self._explainer.train_df.columns,
            )
        self._outcome_table = ExcelTable(
            parent=outcome_frame,
            df=initial_row,
            readonly=True,
            supports_sorting=False,
            supports_query_search=False,
            supports_deletion=False,
            auto_resize=True,
            allow_mouse_clicks=False,
        )
        self._outcome_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._outcome_table.verticalHeader().hide()
        table_height = (
            self._outcome_table.horizontalHeader().sizeHint().height()
            + self._outcome_table.verticalHeader().defaultSectionSize()
            + self._outcome_table.horizontalScrollBar().sizeHint().height()
            + self._outcome_table.frameWidth() * 2
            + 2
        )
        self._outcome_table.setFixedHeight(table_height)
        outcome_layout.addWidget(self._outcome_table)

        self._outcome_plot = RulesPlot(outcome_frame)
        outcome_layout.addWidget(self._outcome_plot, 1)
        self._analysis_layout.addWidget(outcome_frame, 1)

        self._outcome_split_combo.currentTextChanged.connect(self._outcome_split_changed)
        self._outcome_match_spinner.valueChanged.connect(self._update_outcome_explanation)
        self._outcome_plot.show_message('The selected outcome is waiting to render.')

    def _render_initial_tree_views(self):
        """Render tree SVGs only after the dialog has entered its event loop."""

        if self._tree_rendering_blocked or not self.isVisible():
            return
        if self._tree_render_in_progress:
            self._initial_render_timer.start(self.INITIAL_RENDER_DELAY_MS)
            return

        current_rule_tab = self._rule_tabs.currentIndex()
        self._render_rule_tree_once()
        if current_rule_tab != self._rule_tree_index:
            self._rule_tab_changed(current_rule_tab)

        if not self._rule_tree_rendered:
            self._initial_tree_views_rendered = True
            return

        self._update_outcome_explanation()
        self._initial_tree_views_rendered = True

    def _outcome_dataframe(self) -> pd.DataFrame:
        return (
            self._explainer.train_df
            if self._outcome_split_combo.currentText() == 'Train'
            else self._explainer.test_df
        )

    def _outcome_split_changed(self, _value=None):
        dataframe = self._outcome_dataframe()
        self._outcome_match_spinner.blockSignals(True)
        self._outcome_match_spinner.setRange(0, max(len(dataframe) - 1, 0))
        self._outcome_match_spinner.setValue(0)
        self._outcome_match_spinner.setEnabled(not dataframe.empty)
        self._outcome_match_spinner.blockSignals(False)
        self._update_outcome_explanation()

    def _update_outcome_explanation(self, _value=None):
        dataframe = self._outcome_dataframe()
        if dataframe.empty:
            self._outcome_table.modify_rows(
                row_ids=[0],
                items=[[None] * len(self._explainer.train_df.columns)],
            )
            self._outcome_plot.show_message('No matches are available for this dataset.')
            return

        match_index = min(self._outcome_match_spinner.value(), len(dataframe) - 1)
        selected_row = dataframe.iloc[[match_index]]
        self._outcome_table.modify_rows(
            row_ids=[0],
            items=[selected_row.iloc[0].tolist()],
        )

        if self._tree_model is None:
            self._outcome_plot.show_message(
                'The decision-tree model could not be prepared for visualization.'
            )
            return

        render_key = (self._outcome_split_combo.currentText(), match_index)
        if render_key == self._outcome_rendered_key:
            self._outcome_plot.show_tree_page()
            return
        if self._tree_render_in_progress:
            return

        try:
            selected_inputs, _ = self._explainer._dataset_preprocessor.construct_inputs(
                df=selected_row,
                normalizer=self._explainer.normalizer,
            )
            if len(selected_inputs) == 0:
                raise ValueError('The selected match did not produce model inputs.')

            selected_match = selected_inputs[0]
            self._tree_render_in_progress = True
            rendered = self._outcome_plot.show_tree(
                self._tree_model,
                show_progress=True,
                x=selected_match,
            )
            if rendered:
                self._outcome_rendered_key = render_key
            else:
                self._outcome_plot.show_message(
                    'Outcome rendering was cancelled or failed.'
                )
        except Exception as error:
            self._outcome_plot.show_message(
                f'Unable to render the outcome explanation.\n{error}'
            )
        finally:
            self._tree_render_in_progress = False

    def _build_feature_importance_frame(self):
        importance_frame = QGroupBox('Feature Importance', self)
        self._set_analysis_frame_size(importance_frame)
        importance_layout = QVBoxLayout(importance_frame)
        importance_layout.setContentsMargins(12, 12, 12, 12)
        importance_layout.setSpacing(8)

        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel('Sort By Importance', importance_frame))
        self._importance_sort_combo = QComboBox(importance_frame)
        self._importance_sort_combo.addItems(['Descending', 'Ascending'])
        options_layout.addWidget(self._importance_sort_combo)
        options_layout.addStretch(1)
        importance_layout.addLayout(options_layout)

        self._importance_chart = _FeatureImportanceChart(importance_frame)
        importance_layout.addWidget(self._importance_chart, 1)
        self._analysis_layout.addWidget(importance_frame, 1)

        self._importance_sort_combo.currentTextChanged.connect(self._update_importance_chart)
        self._update_importance_chart()

    def _update_importance_chart(self, _value=None):
        self._importance_chart.set_importances(
            importances=self._feature_importances,
            ascending=self._importance_sort_combo.currentText() == 'Ascending',
        )
