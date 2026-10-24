from typing import List, Optional
import numpy as np
import pandas as pd
import shap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PySide6.QtCharts import (QBarCategoryAxis, QBarSet, QChart, QChartView, QHorizontalBarSeries, QValueAxis)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QGroupBox, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QSpinBox, QToolTip, QVBoxLayout, QWidget)
from src.core.interpretability.explainer import ClassifierExplainer
from src.gui.animations.barplot import BarplotScaling
from src.gui.components.dialogs.tasks import SnakeThreadTaskDialog
from src.gui.widgets.table import ExcelTable


def _increase_font(font: QFont, amount: int = 2) -> QFont:
    font = QFont(font)
    if font.pointSize() > 0:
        font.setPointSize(font.pointSize() + amount)
    elif font.pixelSize() > 0:
        font.setPixelSize(font.pixelSize() + amount)
    return font


class _SHAPBarChart(QChartView):
    """Grouped horizontal bar chart comparing train and test SHAP values."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._chart = QChart()
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self._chart.legend().setFont(_increase_font(self._chart.legend().font()))
        self.setChart(self._chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._feature_names = []
        self._barplot_scalings: List[BarplotScaling] = []
        self.show_message('SHAP explanations have not been computed yet.')

    def _reset(self) :
        for animation in self._barplot_scalings:
            animation.stop()
            animation.deleteLater()
        self._barplot_scalings = []

        self._feature_names = []
        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)
            axis.deleteLater()
        self._chart.legend().hide()
        self._chart.setTitle('')

    def show_message(self, message: str) :
        self._reset()
        self._chart.setTitle(message)

    @classmethod
    def _clustered_order(cls, cluster: np.ndarray, train_values: np.ndarray, test_values: np.ndarray) -> List[int]:
        """Keep linked features together while orienting branches by importance."""

        descending_order = np.argsort(-train_values, kind='stable').tolist()
        linkage = np.asarray(cluster)
        feature_count = train_values.size
        importance = (train_values + test_values) / 2.0

        def visit(node_id: int) -> tuple[list[int], float]:
            if node_id < feature_count:
                return [node_id], float(importance[node_id])

            row_index = node_id - feature_count
            if row_index < 0 or row_index >= linkage.shape[0]:
                return [], 0.0

            left_order, left_importance = visit(int(linkage[row_index, 0]))
            right_order, right_importance = visit(int(linkage[row_index, 1]))
            if right_importance > left_importance:
                left_order, right_order = right_order, left_order
                left_importance, right_importance = right_importance, left_importance

            return left_order + right_order, max(left_importance, right_importance)

        order, _ = visit(feature_count + linkage.shape[0] - 1)
        valid_order = []
        seen = set()
        for feature_index in order:
            if 0 <= feature_index < feature_count and feature_index not in seen:
                valid_order.append(feature_index)
                seen.add(feature_index)

        valid_order.extend(index for index in descending_order if index not in seen)
        return valid_order

    def set_values(
            self,
            feature_names: list[str],
            train_values: np.ndarray,
            test_values: np.ndarray,
            cluster: Optional[np.ndarray] = None,
            ascending: bool = False,
    ) :
        self._reset()

        train_values = np.asarray(train_values, dtype=float).reshape(-1)
        test_values = np.asarray(test_values, dtype=float).reshape(-1)
        value_count = min(len(feature_names), train_values.size, test_values.size)
        feature_names = [str(name) for name in feature_names[:value_count]]
        train_values = np.nan_to_num(train_values[:value_count], nan=0.0, posinf=0.0, neginf=0.0)
        test_values = np.nan_to_num(test_values[:value_count], nan=0.0, posinf=0.0, neginf=0.0)
        axis_maximum = max(float(train_values.max()), float(test_values.max()), 0.0)

        if cluster is None:
            if ascending:
                order = np.argsort(train_values, kind='stable').tolist()[:15]
            else:
                order = np.argsort(-train_values, kind='stable').tolist()[:15]
        else:
            order = self._clustered_order(cluster, train_values, test_values)[:15]

        feature_names = [feature_names[index] for index in order]
        train_values = train_values[order]
        test_values = test_values[order]

        # A vertical category axis displays its first entry at the bottom, so
        # reverse the values to keep the most important feature at the top.
        self._feature_names = list(reversed(feature_names))
        display_train_values = train_values[::-1]
        display_test_values = test_values[::-1]

        train_set = QBarSet('Train')
        train_set.append(display_train_values.tolist())
        train_set.setColor(QColor('#FF1744'))
        train_set.setBorderColor(QColor('black'))

        test_set = QBarSet('Test')
        test_set.append(display_test_values.tolist())
        test_set.setColor(QColor('#3D5AFE'))
        test_set.setBorderColor(QColor('black'))

        series = QHorizontalBarSeries()
        series.setBarWidth(0.65)
        series.append(train_set)
        series.append(test_set)
        series.hovered.connect(self._show_tooltip)

        feature_axis = QBarCategoryAxis()
        feature_axis.append(self._feature_names)
        feature_axis.setTitleText('Feature')
        feature_title_font = _increase_font(feature_axis.titleFont())
        feature_title_font.setBold(True)
        feature_axis.setTitleFont(feature_title_font)

        value_axis = QValueAxis()
        value_axis.setTitleText('Mean |SHAP Value|')
        value_title_font = _increase_font(value_axis.titleFont())
        value_title_font.setBold(True)
        value_axis.setTitleFont(value_title_font)
        value_axis.setLabelFormat('%.3f')
        value_axis.setRange(0.0, max(axis_maximum * 1.1, 0.01))
        value_axis.applyNiceNumbers()

        self._chart.addSeries(series)
        self._chart.addAxis(feature_axis, Qt.AlignmentFlag.AlignLeft)
        self._chart.addAxis(value_axis, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(feature_axis)
        series.attachAxis(value_axis)
        chart_title_font = _increase_font(feature_title_font)
        chart_title_font.setBold(True)
        self._chart.setTitleFont(chart_title_font)
        self._chart.setTitle(f'SHAP Importance Analysis of {len(self._feature_names)} Best Features')
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignTop)

        self._barplot_scalings = [
            BarplotScaling(train_set, display_train_values.tolist(), parent=self),
            BarplotScaling(test_set, display_test_values.tolist(), parent=self),
        ]
        for animation in self._barplot_scalings:
            animation.start()

    def _show_tooltip(self, status: bool, index: int, bar_set: QBarSet) :
        if status and 0 <= index < len(self._feature_names):
            QToolTip.showText(
                QCursor.pos(),
                f'{self._feature_names[index]} - {bar_set.label()}: {bar_set.at(index):.6f}',
                self,
            )
        else:
            QToolTip.hideText()


class ExplainerBaseDialog(QDialog):
    """Common window and SHAP explanation UI for model explainer dialogs."""

    ANALYSIS_FRAME_WIDTH = 1050
    ANALYSIS_FRAME_HEIGHT = 560
    ANALYSIS_PAGE_WIDTH = ANALYSIS_FRAME_WIDTH + 32

    def __init__(self, explainer: ClassifierExplainer, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)

        self._explainer = explainer

        self._use_clusters_checkbox = None
        self._shap_sort_label = None
        self._shap_sort_combo = None
        self._shap_chart = None
        self._beeswarm_split_combo = None
        self._beeswarm_target_label = None
        self._beeswarm_target_combo = None
        self._beeswarm_layout = None
        self._beeswarm_image = None
        self._beeswarm_axes = {}
        self._beeswarm_pixmaps = {}
        self._local_split_combo = None
        self._local_match_spinner = None
        self._local_table = None
        self._local_waterfall_image = None
        self._waterfall_pixmaps = {}
        self._analysis_layout = None

        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumSize(800, 600)
        self.resize(900, 520)
        self.setModal(True)

        self._build_ui()
        self._compute_explanations()
        self.build_analysis_frames()

    def _build_ui(self) :
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        dialog_layout.addWidget(scroll_area)

        page = QWidget(scroll_area)
        page.setFixedWidth(self.ANALYSIS_PAGE_WIDTH)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._analysis_layout = layout
        scroll_area.setWidget(page)

        shap_frame = QGroupBox('SHAP Global Explanations', page)
        self._set_analysis_frame_size(shap_frame)
        shap_layout = QVBoxLayout(shap_frame)
        shap_layout.setContentsMargins(12, 12, 12, 12)
        shap_layout.setSpacing(8)

        options_layout = QHBoxLayout()
        self._shap_sort_label = QLabel('Sort By Importance', shap_frame)
        self._shap_sort_combo = QComboBox(shap_frame)
        self._shap_sort_combo.addItems(['Descending', 'Ascending'])
        self._shap_sort_combo.setToolTip('Choose the feature-importance display order.')
        options_layout.addWidget(self._shap_sort_label)
        options_layout.addWidget(self._shap_sort_combo)
        options_layout.addStretch(1)
        self._use_clusters_checkbox = QCheckBox('Use Clusters', shap_frame)
        self._use_clusters_checkbox.setToolTip('Keep related features together using the training feature hierarchy.')
        options_layout.addWidget(self._use_clusters_checkbox)
        shap_layout.addLayout(options_layout)

        self._shap_chart = _SHAPBarChart(shap_frame)
        shap_layout.addWidget(self._shap_chart, 1)
        layout.addWidget(shap_frame, 1)

        beeswarm_frame = QGroupBox('SHAP Beeswarm', page)
        self._set_analysis_frame_size(beeswarm_frame)
        self._beeswarm_layout = QVBoxLayout(beeswarm_frame)
        self._beeswarm_layout.setContentsMargins(12, 12, 12, 12)
        self._beeswarm_layout.setSpacing(8)

        beeswarm_options_layout = QHBoxLayout()
        beeswarm_options_layout.addWidget(QLabel('Dataset', beeswarm_frame))
        self._beeswarm_split_combo = QComboBox(beeswarm_frame)
        self._beeswarm_split_combo.addItems(['Train', 'Test'])
        beeswarm_options_layout.addWidget(self._beeswarm_split_combo)
        self._beeswarm_target_label = QLabel('Target', beeswarm_frame)
        self._beeswarm_target_combo = QComboBox(beeswarm_frame)
        self._beeswarm_target_label.hide()
        self._beeswarm_target_combo.hide()
        beeswarm_options_layout.addSpacing(16)
        beeswarm_options_layout.addWidget(self._beeswarm_target_label)
        beeswarm_options_layout.addWidget(self._beeswarm_target_combo)
        beeswarm_options_layout.addStretch(1)
        self._beeswarm_layout.addLayout(beeswarm_options_layout)

        self._beeswarm_image = QLabel('SHAP explanations have not been computed yet.', beeswarm_frame)
        self._beeswarm_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._beeswarm_image.setScaledContents(True)
        self._beeswarm_image.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._beeswarm_layout.addWidget(self._beeswarm_image, 1)
        layout.addWidget(beeswarm_frame, 1)

        self._build_local_explanations_frame(page)

        self._shap_sort_combo.currentTextChanged.connect(self._update_shap_chart)
        self._use_clusters_checkbox.toggled.connect(self._use_clusters_changed)
        self._beeswarm_split_combo.currentTextChanged.connect(self._update_beeswarm_plot)
        self._beeswarm_target_combo.currentIndexChanged.connect(self._update_beeswarm_plot)
        self._beeswarm_target_combo.currentIndexChanged.connect(self._update_local_explanation)

    def _build_local_explanations_frame(self, parent: QWidget) :
        local_frame = QGroupBox('SHAP Local Explanations', parent)
        self._set_analysis_frame_size(local_frame)
        local_layout = QVBoxLayout(local_frame)
        local_layout.setContentsMargins(12, 12, 12, 12)
        local_layout.setSpacing(8)

        self._local_match_spinner = QSpinBox(local_frame)
        self._local_match_spinner.setFixedWidth(100)
        self._local_match_spinner.setRange(0, max(len(self._explainer.train_df) - 1, 0))
        self._local_match_spinner.setValue(0)
        self._local_match_spinner.setEnabled(not self._explainer.train_df.empty)

        explanation_layout = QVBoxLayout()
        explanation_layout.setSpacing(8)
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel('Dataset', local_frame))
        self._local_split_combo = QComboBox(local_frame)
        self._local_split_combo.addItems(['Train', 'Test'])
        dataset_layout.addWidget(self._local_split_combo)
        dataset_layout.addSpacing(16)
        dataset_layout.addWidget(QLabel('Match ID', local_frame))
        dataset_layout.addWidget(self._local_match_spinner)
        dataset_layout.addStretch(1)
        explanation_layout.addLayout(dataset_layout)

        initial_row = self._explainer.train_df.head(1).reset_index(drop=True)
        if initial_row.empty:
            initial_row = pd.DataFrame(
                [[None] * len(self._explainer.train_df.columns)],
                columns=self._explainer.train_df.columns,
            )
        self._local_table = ExcelTable(
            parent=local_frame,
            df=initial_row,
            readonly=True,
            supports_sorting=False,
            supports_query_search=False,
            supports_deletion=False,
            auto_resize=True,
            allow_mouse_clicks=False,
        )
        self._local_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._local_table.verticalHeader().hide()
        table_height = (
            self._local_table.horizontalHeader().sizeHint().height()
            + self._local_table.verticalHeader().defaultSectionSize()
            + self._local_table.horizontalScrollBar().sizeHint().height()
            + self._local_table.frameWidth() * 2
            + 2
        )
        self._local_table.setFixedHeight(table_height)
        explanation_layout.addWidget(self._local_table)

        self._local_waterfall_image = QLabel(
            'Select a match to view its local SHAP explanation.',
            local_frame,
        )
        self._local_waterfall_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._local_waterfall_image.setScaledContents(True)
        self._local_waterfall_image.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        explanation_layout.addWidget(self._local_waterfall_image, 1)
        local_layout.addLayout(explanation_layout, 1)
        self._analysis_layout.addWidget(local_frame, 1)

        self._local_split_combo.currentTextChanged.connect(self._local_split_changed)
        self._local_match_spinner.valueChanged.connect(self._update_local_explanation)

    def _set_analysis_frame_size(self, frame: QWidget) :
        """Keep plots independent of dialog resize events."""

        frame.setFixedSize(self.ANALYSIS_FRAME_WIDTH, self.ANALYSIS_FRAME_HEIGHT)

    def build_analysis_frames(self) :
        """Allow concrete explainer dialogs to append model-specific frames."""

        pass

    def _use_clusters_changed(self, checked: bool) :
        self._shap_sort_label.setEnabled(not checked)
        self._shap_sort_combo.setEnabled(not checked)
        self._update_shap_chart()

    def explainer_model(self) :
        """Compute model explanations; subclasses may extend this task hook."""

        self._explainer.compute_shap_values()
        self._generate_beeswarm_axes()

    def _generate_beeswarm_axes(self) :
        """Build beeswarm axes in the model-explanation worker task."""

        self._beeswarm_axes = {}
        explanations = {
            'Train': self._explainer.train_shap_values,
            'Test': self._explainer.eval_shap_values,
        }
        for split, explanation in explanations.items():
            values = np.asarray(explanation.values)
            target_indices = range(values.shape[-1]) if values.ndim > 2 else [None]
            for target_index in target_indices:
                target_explanation = (
                    explanation[..., target_index]
                    if target_index is not None
                    else explanation
                )
                figure = Figure(figsize=(10, 5), constrained_layout=True)
                plot_axis = figure.add_subplot(111)
                self._beeswarm_axes[(split, target_index)] = shap.plots.beeswarm(
                    target_explanation,
                    max_display=15,
                    ax=plot_axis,
                    show=False,
                    plot_size=None,
                    group_remaining_features=False,
                )

    def _compute_explanations(self) :
        task = SnakeThreadTaskDialog(
            self.explainer_model,
            title='Computing Model Explanations',
            description='Computing model explanations. Please wait...',
            parent=self,
        )
        task.run()
        task.deleteLater()

        self._update_shap_chart()
        self._configure_beeswarm_targets()
        self._update_beeswarm_plot()
        self._update_local_explanation()

    @staticmethod
    def _mean_absolute_shap_values(explanation, feature_count: int) -> np.ndarray:
        values = np.asarray(explanation.values, dtype=float)
        candidate_axes = [axis for axis, size in enumerate(values.shape) if size == feature_count]
        feature_axis = 1 if 1 in candidate_axes else candidate_axes[0]
        values = np.moveaxis(values, feature_axis, -1)
        reduction_axes = tuple(range(values.ndim - 1))
        if reduction_axes:
            values = np.nanmean(np.abs(values), axis=reduction_axes)
        else:
            values = np.abs(values)
        return np.asarray(values, dtype=float).reshape(-1)


    def _update_shap_chart(self, _checked: Optional[bool] = None) :
        feature_names = self._explainer.feature_names
        num_features = len(feature_names)
        train_values = self._mean_absolute_shap_values(self._explainer.train_shap_values, num_features)
        test_values = self._mean_absolute_shap_values(self._explainer.eval_shap_values, num_features)
        cluster = self._explainer.train_cluster if self._use_clusters_checkbox.isChecked() else None
        self._shap_chart.set_values(
            feature_names=feature_names,
            train_values=train_values,
            test_values=test_values,
            cluster=cluster,
            ascending=self._shap_sort_combo.currentText() == 'Ascending',
        )

    def _configure_beeswarm_targets(self) :
        values = np.asarray(self._explainer.train_shap_values.values)
        output_count = values.shape[-1] if values.ndim > 2 else 0

        self._beeswarm_target_combo.blockSignals(True)
        self._beeswarm_target_combo.clear()
        for target_index in range(output_count):
            target_label = (
                self._explainer.target_names[target_index]
                if target_index < len(self._explainer.target_names)
                else f'Target {target_index + 1}'
            )
            self._beeswarm_target_combo.addItem(str(target_label), userData=target_index)
        self._beeswarm_target_combo.blockSignals(False)

        has_multiple_outputs = output_count > 1
        self._beeswarm_target_label.setVisible(has_multiple_outputs)
        self._beeswarm_target_combo.setVisible(has_multiple_outputs)

    def _update_beeswarm_plot(self, _value=None) :
        split = self._beeswarm_split_combo.currentText()
        explanation = (
            self._explainer.train_shap_values
            if split == 'Train'
            else self._explainer.eval_shap_values
        )
        target_index = None
        if np.asarray(explanation.values).ndim > 2:
            target_index = self._beeswarm_target_combo.currentData()
            target_index = 0 if target_index is None else target_index

        plot_key = (split, target_index)
        cached_pixmap = self._beeswarm_pixmaps.get(plot_key)
        if cached_pixmap is not None:
            self._beeswarm_image.setPixmap(cached_pixmap)
            return

        beeswarm_axis = self._beeswarm_axes[plot_key]

        # Matplotlib canvases redraw their complete plot on every resize event,
        # which is particularly expensive for a beeswarm. Render it once and
        # let Qt scale the resulting pixmap while the dialog is resized.
        pixmap = self._figure_to_pixmap(beeswarm_axis.figure)
        self._beeswarm_pixmaps[plot_key] = pixmap
        self._beeswarm_image.setPixmap(pixmap)

    @staticmethod
    def _figure_to_pixmap(figure: Figure) -> QPixmap:
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        height, width = rgba.shape[:2]
        image = QImage(
            rgba.data,
            width,
            height,
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        return QPixmap.fromImage(image.copy())

    def _local_split_changed(self, _value=None) :
        dataframe = self._local_dataframe()
        self._local_match_spinner.blockSignals(True)
        self._local_match_spinner.setRange(0, max(len(dataframe) - 1, 0))
        self._local_match_spinner.setValue(0)
        self._local_match_spinner.setEnabled(not dataframe.empty)
        self._local_match_spinner.blockSignals(False)
        self._update_local_explanation()

    def _local_dataframe(self) -> pd.DataFrame:
        return (
            self._explainer.train_df
            if self._local_split_combo.currentText() == 'Train'
            else self._explainer.test_df
        )

    def _update_local_explanation(self, _value=None) :
        dataframe = self._local_dataframe()
        if dataframe.empty:
            self._local_waterfall_image.clear()
            self._local_waterfall_image.setText('No matches are available for this dataset.')
            return

        match_index = min(self._local_match_spinner.value(), len(dataframe) - 1)
        self._local_table.modify_rows(
            row_ids=[0],
            items=[dataframe.iloc[match_index].tolist()],
        )

        split = self._local_split_combo.currentText()
        explanation = (
            self._explainer.train_shap_values
            if split == 'Train'
            else self._explainer.eval_shap_values
        )
        target_index = None
        if np.asarray(explanation.values).ndim > 2:
            target_index = self._beeswarm_target_combo.currentData()
            target_index = 0 if target_index is None else target_index
            local_explanation = explanation[match_index, :, target_index]
        else:
            local_explanation = explanation[match_index]

        plot_key = (split, target_index, match_index)
        cached_pixmap = self._waterfall_pixmaps.get(plot_key)
        if cached_pixmap is None:
            plt.figure(figsize=(10, 4))
            waterfall_axis = shap.plots.waterfall(
                local_explanation,
                max_display=15,
                show=False,
            )
            waterfall_figure = waterfall_axis.figure
            waterfall_figure.set_size_inches(10, 4)
            cached_pixmap = self._figure_to_pixmap(waterfall_figure)
            plt.close(waterfall_figure)

            # Keep recently viewed samples fast without retaining an image for
            # every match in a potentially large league dataset.
            if len(self._waterfall_pixmaps) >= 32:
                oldest_key = next(iter(self._waterfall_pixmaps))
                self._waterfall_pixmaps.pop(oldest_key)
            self._waterfall_pixmaps[plot_key] = cached_pixmap

        self._local_waterfall_image.setPixmap(cached_pixmap)
