import numpy as np
import pandas as pd
from typing import Optional
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.interpretability.explainers.lda import DiscriminantAnalysisExplainer, DiscriminantPlotData
from src.gui.components.dialogs.models.interpretability.explainers.base import ExplainerBaseDialog
from src.gui.components.dialogs.models.interpretability.explainers.logistic import _CoefficientChart

_CLASS_COLORS = np.asarray(
    [
        (220, 65, 65),
        (55, 135, 205),
        (55, 165, 95),
        (236, 146, 42),
        (139, 91, 176),
    ],
    dtype=float,
) / 255.0
_BOUNDARY_COLOR = '#4E185C'
_MEAN_COLOR = '#FFD700'


def _class_color(class_index: int) -> np.ndarray:
    return _CLASS_COLORS[class_index % len(_CLASS_COLORS)]


def _sample_colors(probabilities: np.ndarray, mask: np.ndarray, class_index: int) -> np.ndarray:
    """Tint samples by the model's confidence in their true class."""

    probabilities = np.asarray(probabilities, dtype=float)
    confidence = probabilities[mask, class_index]
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0)
    chance = 1.0 / probabilities.shape[1]
    normalized = np.clip((confidence - chance) / max(1e-12, 1.0 - chance), 0.0, 1.0)
    # Match the test visualization: uncertain points remain visible but
    # approach white, while confident points approach their class color.
    strength = (0.25 + 0.75 * normalized)[:, None]
    return (1.0 - strength) + strength * _class_color(class_index)[None, :]


def _render_transformed_boundaries(data: DiscriminantPlotData) -> np.ndarray:
    """Render the transformed-space visualization without constructing Qt objects."""

    figure = Figure(figsize=(10.0, 4.8), dpi=110, constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)

    x_min, x_max = (float(value) for value in data.x_range)
    y_min, y_max = (float(value) for value in data.y_range)
    background = np.asarray(data.background_rgba, dtype=np.uint8)
    if background.size:
        axis.imshow(
            background,
            extent=(x_min, x_max, y_min, y_max),
            origin='upper' if not data.is_1d else 'lower',
            interpolation='bilinear',
            aspect='auto',
            zorder=0,
        )

    z = np.asarray(data.z, dtype=float)
    y_true = np.asarray(data.y_true)
    sample_probabilities = np.asarray(data.sample_probabilities, dtype=float)
    means = np.asarray(data.means, dtype=float)
    class_values = list(data.class_values)
    class_names = [str(name) for name in data.class_names]

    if data.is_1d:
        gaussian_x = np.asarray(data.gaussian_x, dtype=float)
        gaussian_pdfs = np.asarray(data.gaussian_pdfs, dtype=float)
        for class_index, class_name in enumerate(class_names):
            color = _class_color(class_index)
            if class_index < len(gaussian_pdfs):
                axis.plot(
                    gaussian_x,
                    gaussian_pdfs[class_index],
                    color=color,
                    linewidth=2.2,
                    label=f'{class_name} Gaussian',
                    zorder=3
                )

            if class_index < len(means):
                mean_x = float(means[class_index, 0])
                axis.axvline(
                    mean_x,
                    color=color * 0.72,
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.95,
                    zorder=2
                )
                axis.scatter(
                    [mean_x],
                    [0.0],
                    marker='*',
                    s=190,
                    facecolor=_MEAN_COLOR,
                    edgecolor='#2D2D2D',
                    linewidth=0.9,
                    zorder=7
                )

        for boundary_index, boundary_x in enumerate(data.boundary_positions):
            axis.axvline(
                float(boundary_x),
                color=_BOUNDARY_COLOR,
                linewidth=2.0,
                label='Decision boundary' if boundary_index == 0 else None,
                zorder=5
            )

        sample_y = np.zeros(len(z), dtype=float)
        x_values = z[:, 0]
        axis.set_ylabel('Gaussian density - samples on y = 0')
    else:
        for polyline_index, polyline in enumerate(data.boundary_polylines):
            points = np.asarray(polyline, dtype=float)
            if points.ndim != 2 or points.shape[0] < 2:
                continue
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=_BOUNDARY_COLOR,
                linewidth=2.0,
                label='Decision boundary' if polyline_index == 0 else None,
                zorder=4
            )

        for class_index, ellipse in enumerate(data.ellipses):
            points = np.asarray(ellipse, dtype=float)
            if points.ndim != 2 or points.shape[0] < 2:
                continue
            class_name = (
                class_names[class_index]
                if class_index < len(class_names)
                else f'Class {class_index}'
            )
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=_class_color(class_index) * 0.72,
                linewidth=1.8,
                label=f'{class_name} 2-sigma covariance',
                zorder=3
            )

        for class_index in range(min(len(means), len(class_names))):
            axis.scatter(
                [means[class_index, 0]],
                [means[class_index, 1]],
                marker='*',
                s=190,
                facecolor=_MEAN_COLOR,
                edgecolor='#2D2D2D',
                linewidth=0.9,
                zorder=7
            )

        x_values = z[:, 0]
        sample_y = z[:, 1]
        axis.set_ylabel('LD2')

    for class_index, class_value in enumerate(class_values):
        if class_index >= sample_probabilities.shape[1]:
            break
        mask = y_true == class_value
        if not np.any(mask):
            continue
        class_name = class_names[class_index] if class_index < len(class_names) else str(class_value)
        axis.scatter(
            x_values[mask],
            sample_y[mask],
            s=24,
            c=_sample_colors(sample_probabilities, mask, class_index),
            edgecolors=[_class_color(class_index) * 0.62],
            linewidths=0.55,
            label=f'{class_name} samples',
            zorder=6
        )

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel('LD1')
    axis.set_title(str(data.title), fontweight='bold')
    axis.grid(True, color='#666666', linewidth=0.5, alpha=0.18, zorder=1)

    handles, labels = axis.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label and label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    if unique_handles:
        axis.legend(
            unique_handles,
            unique_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.13),
            ncol=min(4, len(unique_handles)),
            fontsize=8,
            frameon=True
        )
    canvas.draw()
    return np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()


class DiscriminantAnalysisExplainerDialog(ExplainerBaseDialog):
    """Dialog for interpreting linear and quadratic discriminant models."""

    def __init__(self, df: pd.DataFrame, model_id: str, model_database: ModelDatabase, parent: Optional[QWidget] = None):
        self._coefficients = None
        self._coefficient_sort_combo = None
        self._coefficient_target_combo = None
        self._coefficient_chart = None
        self._transformed_plot_data = None
        self._transformed_plot_rgba = None
        self._transformed_plot_image = None

        explainer = DiscriminantAnalysisExplainer(df=df, model_id=model_id, database=model_database)
        self._is_linear = explainer.model_config.classifier_config.get('decision_boundary') == 'linear'

        super().__init__(explainer=explainer, parent=parent)
        self.setWindowTitle(model_id)

    def explainer_model(self):
        """Prepare SHAP and transformed-space rendering in the worker task."""

        super().explainer_model()
        if self._is_linear:
            self._coefficients = self._explainer.get_coefficients()

        self._transformed_plot_data = self._explainer.get_transformed_plot_data()
        self._transformed_plot_rgba = _render_transformed_boundaries(self._transformed_plot_data)

    def build_analysis_frames(self):
        if self._is_linear:
            self._build_coefficients_frame()
        self._build_transformed_boundaries_frame()

    def _build_coefficients_frame(self):
        coefficients_frame = QGroupBox('Coefficients Analysis', self)
        coefficients_frame.setObjectName('ldaCoefficientsFrame')
        self._set_analysis_frame_size(coefficients_frame)
        coefficients_layout = QVBoxLayout(coefficients_frame)
        coefficients_layout.setContentsMargins(12, 12, 12, 12)
        coefficients_layout.setSpacing(8)

        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel('Sort By Importance', coefficients_frame))
        self._coefficient_sort_combo = QComboBox(coefficients_frame)
        self._coefficient_sort_combo.addItems(['Descending', 'Ascending'])
        options_layout.addWidget(self._coefficient_sort_combo)

        if self._coefficients is not None and not {'Feature', 'Coefficient'}.issubset(self._coefficients.columns):
            options_layout.addSpacing(16)
            options_layout.addWidget(QLabel('Target', coefficients_frame))
            self._coefficient_target_combo = QComboBox(coefficients_frame)
            for target in self._coefficients.index:
                self._coefficient_target_combo.addItem(str(target), userData=target)
            options_layout.addWidget(self._coefficient_target_combo)

        options_layout.addStretch(1)
        coefficients_layout.addLayout(options_layout)

        self._coefficient_chart = _CoefficientChart(coefficients_frame)
        coefficients_layout.addWidget(self._coefficient_chart, 1)
        self._analysis_layout.addWidget(coefficients_frame, 1)

        self._coefficient_sort_combo.currentTextChanged.connect(self._update_coefficient_chart)
        if self._coefficient_target_combo is not None:
            self._coefficient_target_combo.currentIndexChanged.connect(self._update_coefficient_chart)
        self._update_coefficient_chart()

    def _update_coefficient_chart(self, _value=None):
        if self._coefficients is None:
            return
        target = self._coefficient_target_combo.currentData() if self._coefficient_target_combo is not None else None
        self._coefficient_chart.set_coefficients(
            coefficients=self._coefficients,
            target=target,
            ascending=self._coefficient_sort_combo.currentText() == 'Ascending',
        )

    def _build_transformed_boundaries_frame(self):
        boundary_frame = QGroupBox('Transformed Decision Boundaries', self)
        boundary_frame.setObjectName('ldaTransformedBoundariesFrame')
        self._set_analysis_frame_size(boundary_frame)
        boundary_layout = QVBoxLayout(boundary_frame)
        boundary_layout.setContentsMargins(12, 12, 12, 12)

        self._transformed_plot_image = QLabel(boundary_frame)
        self._transformed_plot_image.setObjectName('ldaTransformedBoundariesImage')
        self._transformed_plot_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._transformed_plot_image.setScaledContents(True)
        self._transformed_plot_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._transformed_plot_image.setPixmap(self._rgba_to_pixmap(self._transformed_plot_rgba))
        boundary_layout.addWidget(self._transformed_plot_image, 1)
        self._analysis_layout.addWidget(boundary_frame, 1)

    @staticmethod
    def _rgba_to_pixmap(rgba: np.ndarray) -> QPixmap:
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        height, width = rgba.shape[:2]
        image = QImage(
            rgba.data,
            width,
            height,
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        return QPixmap.fromImage(image.copy())
