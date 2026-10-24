import os
import matplotlib as mpl
import numpy as np
import pandas as pd
from collections.abc import Mapping
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Optional
from unittest.mock import patch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QUrl, Qt
from PySide6.QtGui import QColor, QMouseEvent, QWheelEvent
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget
)
from src.gui.components.dialogs.tasks import Task


def _render_tree_svg(
        tree_model: Any,
        filepath: str,
        background: str,
        x: Optional[pd.Series] = None,
        view_options: Optional[Dict[str, Any]] = None,
) -> str:
    """Render and prepare one dtreeviz SVG for the embedded browser."""

    options = {'scale': 1.0, 'x': x}
    if view_options:
        options.update(view_options)
    with mpl.rc_context({'axes.labelweight': 'bold'}):
        render = tree_model.view(**options)
    render.save(filepath)

    tree_path = Path(filepath)
    if not tree_path.exists():
        raise FileNotFoundError(f'dtreeviz did not create {tree_path}.')

    svg = tree_path.read_text(encoding='utf-8')
    svg = svg.replace('fill="white" stroke="none"', f'fill="{background}" stroke="none"', 1)
    svg = svg.replace('#ffffff', background)
    svg = svg.replace('#FFFFFF', background)
    tree_path.write_text(svg, encoding='utf-8')
    return str(tree_path)


class _ZoomableTreeView(QWebEngineView):
    """QWebEngineView with explicit button and Ctrl+wheel zoom support."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._zoom_step = 1.25
        self._minimum_zoom = 0.25
        self._maximum_zoom = 5.0

        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)

    def zoom_in(self):
        self.setZoomFactor(min(self.zoomFactor() * self._zoom_step, self._maximum_zoom))

    def zoom_out(self):
        self.setZoomFactor(max(self.zoomFactor() / self._zoom_step, self._minimum_zoom))

    def zoom_from_wheel(self, event: QWheelEvent) -> bool:
        delta = event.angleDelta().y()

        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()
        else:
            return False
        return True

    def eventFilter(self, watched, event):
        belongs_to_tree = (
            isinstance(watched, QWidget)
            and (watched is self or self.isAncestorOf(watched))
        )
        if (
                belongs_to_tree
                and event.type() == QEvent.Type.Wheel
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier
                and self.zoom_from_wheel(event)
        ):
            event.accept()
            return True
        return super().eventFilter(watched, event)


class _TreePanOverlay(QWidget):
    """ Transparent mouse layer that pans a QWebEngineView by dragging. """

    def __init__(self, tree: _ZoomableTreeView, parent: Optional[QWidget] = None,):
        super().__init__(parent)

        self._tree = tree
        self._last_position = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.hide()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_position = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_position is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._last_position
            self._last_position = event.position()
            zoom = max(self._tree.zoomFactor(), 0.25)
            self._tree.page().runJavaScript(f'window.scrollBy({-delta.x() / zoom}, {-delta.y() / zoom});')
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_position = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and self._tree.zoom_from_wheel(event):
            event.accept()
            return

        pixel_delta = event.pixelDelta()

        if pixel_delta.isNull():
            pixel_delta = event.angleDelta() / 2

        zoom = max(self._tree.zoomFactor(), 0.25)
        self._tree.page().runJavaScript(f'window.scrollBy({-pixel_delta.x() / zoom}, {-pixel_delta.y() / zoom});')
        event.accept()


class RulesPlot(QStackedWidget):
    """Display dtreeviz SVG and Matplotlib visualizations in one plot area."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._background = '#CFD8DC'

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._tree = _ZoomableTreeView(self)
        self._tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._tree.setStyleSheet(f'background-color: {self._background};')

        settings = self._tree.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        self._tree.page().setBackgroundColor(QColor(self._background))
        self._tree_page = QWidget(self)
        self._tree_page.setStyleSheet(f'background-color: {self._background};')

        tree_layout = QVBoxLayout(self._tree_page)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.addWidget(self._tree)

        self._pan_overlay = _TreePanOverlay(tree=self._tree, parent=self._tree_page)
        self._tree_controls = QFrame(self._tree_page)
        self._tree_controls.setObjectName('treeControls')
        self._tree_controls.setStyleSheet(
            f'''
            QFrame#treeControls {{
                background-color: transparent;
                border: none;
            }}
            QToolButton {{
                background-color: #0d6efd;
                color: white;
                border: 1px solid #0d6efd;
                border-radius: 3px;
                padding: 4px 7px;
            }}
            QToolButton:hover {{
                background-color: #2684ff;
                border-color: #2684ff;
            }}
            QToolButton:pressed, QToolButton:checked {{
                background-color: #0b5ed7;
                border-color: #0b5ed7;
            }}
            '''
        )

        controls_layout = QHBoxLayout(self._tree_controls)
        controls_layout.setContentsMargins(1, 1, 1, 1)
        controls_layout.setSpacing(3)

        self.zoom_in_button = QToolButton(self._tree_controls)
        self.zoom_in_button.setText('Zoom +')
        self.zoom_in_button.setToolTip('Zoom in')
        self.zoom_in_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.zoom_in_button.clicked.connect(self._zoom_in)
        controls_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QToolButton(self._tree_controls)
        self.zoom_out_button.setText('Zoom -')
        self.zoom_out_button.setToolTip('Zoom out')
        self.zoom_out_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.zoom_out_button.clicked.connect(self._zoom_out)
        controls_layout.addWidget(self.zoom_out_button)

        self.move_button = QToolButton(self._tree_controls)
        self.move_button.setText('Move')
        self.move_button.setToolTip('Move inside the tree plot')
        self.move_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.move_button.setCheckable(True)
        self.move_button.toggled.connect(self._set_move_enabled)
        controls_layout.addWidget(self.move_button)

        self._position_tree_overlays()
        self._tree_path = Path(gettempdir()) / (f'dtreeviz_qt_{os.getpid()}_{id(self)}.svg')
        self._figure = Figure(facecolor=self._background)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._message = QLabel(self)
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)

        self.addWidget(self._tree_page)
        self.addWidget(self._canvas)
        self.addWidget(self._message)

        self.show_message('No rule-extraction results are available.')

    def show_message(self, message: str):
        """Display a centered message."""

        self._message.setText(message)
        self.setCurrentWidget(self._message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_tree_overlays()

    def _position_tree_overlays(self):
        """ Keep the pan layer and mode buttons over the tree viewport. """

        if not hasattr(self, '_tree_page'):
            return

        self._pan_overlay.setGeometry(self._tree_page.rect())
        self._tree_controls.adjustSize()
        margin = 8
        self._tree_controls.move(max(margin, self._tree_page.width() - self._tree_controls.width() - margin), margin)
        self._tree_controls.raise_()

    def _zoom_in(self):
        """Increase the tree browser's zoom by one step."""

        self._tree.zoom_in()

    def _zoom_out(self):
        """Decrease the tree browser's zoom by one step."""

        self._tree.zoom_out()

    def _set_move_enabled(self, enabled: bool):
        """Enable or disable click-and-drag movement inside the tree."""

        self._pan_overlay.setVisible(enabled)

        if enabled:
            self._pan_overlay.raise_()

        self._tree_controls.raise_()

    def show_tree(
            self,
            tree_model: Any,
            show_progress: bool = False,
            x: Any = None,
            view_options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Render a complete tree or one selected observation path."""

        arguments = (
            tree_model,
            str(self._tree_path),
            self._background,
            x,
            view_options,
        )
        if show_progress:
            task = Task(
                _render_tree_svg,
                *arguments,
                title='Rendering Tree Model',
                description='Please wait while the tree model is rendered.',
                parent=self,
            )
            filepath = task.run()
            if filepath is None:
                return False
        else:
            filepath = _render_tree_svg(*arguments)

        url = QUrl.fromLocalFile(filepath)

        if self._tree.url() == url:
            self._tree.reload()
        else:
            self._tree.setUrl(url)

        self.setCurrentWidget(self._tree_page)
        return True

    def show_tree_page(self):
        """Show the previously rendered tree without re-rendering it."""

        self.setCurrentWidget(self._tree_page)

    def show_matplotlib_page(self):
        """Show the existing Matplotlib plot without redrawing it."""

        self.setCurrentWidget(self._canvas)

    def _new_axis(self):
        """Create a fresh Matplotlib axis."""

        self._figure.clear()
        self._figure.set_facecolor(self._background)

        axis = self._figure.add_subplot(111)
        axis.set_facecolor(self._background)
        return axis

    def _finish_matplotlib_plot(self):
        """Finish and display the current Matplotlib plot."""

        self._figure.tight_layout()
        self._canvas.draw()
        self.setCurrentWidget(self._canvas)

    def show_leaf_distributions(self, tree_model: Any):
        """Display dtreeviz class distributions for all leaves."""

        try:
            axis = self._new_axis()

            # dtreeviz calls pyplot.show() even when an axis is supplied,
            # which can make IDE backends open a second blank canvas.
            with patch('matplotlib.pyplot.show'):
                tree_model.ctree_leaf_distributions(ax=axis)

            axis.set_facecolor(self._background)
            axis.set_title('Leaf Distributions of the Decision Tree', fontweight='bold')
            axis.xaxis.label.set_fontweight('bold')
            axis.yaxis.label.set_fontweight('bold')

            legend = axis.get_legend()

            if legend is not None:
                legend.get_frame().set_facecolor(self._background)

            self._finish_matplotlib_plot()

        except Exception as error:
            self.show_message(
                f'Unable to render leaf distributions.\n{error}'
            )

    def show_leaf_distribution_values(
            self,
            leaf_ids: np.ndarray,
            class_counts: np.ndarray,
            class_names: list[Any],
            total_leaves: int,
    ):
        """Plot precomputed distributions for a bounded set of leaves."""

        try:
            axis = self._new_axis()
            positions = np.arange(len(leaf_ids))
            bottoms = np.zeros(len(leaf_ids), dtype=float)
            for class_index, class_name in enumerate(class_names):
                values = class_counts[:, class_index]
                axis.bar(
                    positions,
                    values,
                    bottom=bottoms,
                    label=str(class_name),
                )
                bottoms += values

            axis.set_xticks(positions)
            axis.set_xticklabels(
                [str(leaf_id) for leaf_id in leaf_ids],
                rotation=90,
            )
            shown = len(leaf_ids)
            suffix = (
                f' (showing {shown} of {total_leaves})'
                if shown < total_leaves else ''
            )
            axis.set_title(
                f'Leaf Distributions of the Decision Tree{suffix}',
                fontweight='bold',
            )
            axis.set_xlabel('Leaf ID', fontweight='bold')
            axis.set_ylabel('Samples', fontweight='bold')
            if shown:
                legend = axis.legend()
                legend.get_frame().set_facecolor(self._background)
            self._finish_matplotlib_plot()
        except Exception as error:
            self.show_message(
                f'Unable to render leaf distributions.\n{error}'
            )

    def show_feature_space(self, tree_model: Any, features: list[Any]):
        """Display one/two-dimensional dtreeviz feature space."""

        try:
            axis = self._new_axis()
            tree_model.ctree_feature_space(
                features=features,
                show={'title', 'legend', 'splits'},
                ax=axis,
            )
            axis.set_facecolor(self._background)

            for patch in axis.patches:
                patch.set_zorder(0)

                alpha = patch.get_alpha()
                if alpha is not None:
                    patch.set_alpha(min(alpha, 0.20))

            # Scatter observations above the regions.
            for collection in axis.collections:
                collection.set_zorder(3)

            # Split lines above both.
            for line in axis.lines:
                line.set_zorder(4)

            if len(features) == 2:
                axis.set_box_aspect(1)

            axis.xaxis.label.set_fontweight('bold')
            axis.yaxis.label.set_fontweight('bold')

            legend = axis.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(self._background)

            self._finish_matplotlib_plot()

        except Exception as error:
            self.show_message(f'Unable to render the feature space.\n{error}')


class RulesReport(QFrame):
    """Interactive rule-extraction report for target-specific tree models."""

    @staticmethod
    def has_results(rules_dict: Optional[Dict[str, Any]], **_: Any) -> bool:
        """Return whether rule extraction completed successfully."""

        return rules_dict is not None

    def __init__(self, rules_dict: Optional[Dict[str, Any]], parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._initializing = True
        self._rules_dict = dict(rules_dict) if isinstance(rules_dict, Mapping) else {}

        font = self.font()
        font.setPointSize(font.pointSize() + 2)
        self.setFont(font)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(900, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        title = QLabel('Rule Extraction', self)
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 2)
        title_font.setBold(True)
        title.setFont(title_font)

        self.mode_tabs = QTabBar(self)
        self.mode_tabs.setExpanding(False)
        self.mode_tabs.setDrawBase(False)
        self.tree_index = self.mode_tabs.addTab('Tree Model')
        self.leaf_index = self.mode_tabs.addTab('Leaf Distributions')
        self.feature_space_index = self.mode_tabs.addTab('Feature Space')
        tab_spacing = QWidget(self.mode_tabs)
        tab_spacing.setFixedWidth(12)
        tab_spacing.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.mode_tabs.setTabButton(self.tree_index, QTabBar.ButtonPosition.RightSide, tab_spacing)

        self.target_label = QLabel('Target:', self)
        self.target_combo = QComboBox(self)
        self.target_combo.setMinimumWidth(110)
        self.target_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.target_combo.setMinimumContentsLength(10)

        self.feature_1_label = QLabel('Feature 1:', self)
        self.feature_1_combo = QComboBox(self)
        self.feature_1_combo.setMinimumWidth(85)
        self.feature_1_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.feature_1_combo.setMinimumContentsLength(8)

        self.feature_2_label = QLabel('Feature 2:', self)
        self.feature_2_combo = QComboBox(self)
        self.feature_2_combo.setMinimumWidth(85)
        self.feature_2_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.feature_2_combo.setMinimumContentsLength(8)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        toolbar.addWidget(self.mode_tabs)
        toolbar.addStretch(1)
        toolbar.addWidget(self.target_label)
        toolbar.addWidget(self.target_combo)
        toolbar.addWidget(self.feature_1_label)
        toolbar.addWidget(self.feature_1_combo)
        toolbar.addWidget(self.feature_2_label)
        toolbar.addWidget(self.feature_2_combo)

        self.plot = RulesPlot(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addLayout(toolbar)
        layout.addWidget(self.plot, 1)

        self._populate_targets()
        self.mode_tabs.currentChanged.connect(self._mode_changed)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.feature_1_combo.currentIndexChanged.connect(self._update_plot)
        self.feature_2_combo.currentIndexChanged.connect(self._update_plot)
        self._update_feature_visibility()
        self._update_plot()
        self._initializing = False

    def _populate_targets(self):
        self.target_combo.blockSignals(True)
        self.target_combo.clear()

        for target in self._rules_dict:
            self.target_combo.addItem(str(target), userData=target)

        self.target_combo.setCurrentIndex(0 if self._rules_dict else -1)
        self.target_combo.setEnabled(bool(self._rules_dict))
        self.target_combo.blockSignals(False)
        self._populate_features()

    def _current_tree_model(self) -> tuple[Any, Any]:
        index = self.target_combo.currentIndex()
        if index < 0:
            return None, None
        target = self.target_combo.itemData(index)
        return target, self._rules_dict.get(target)

    @staticmethod
    def _feature_names(tree_model: Any) -> list[Any]:
        shadow_tree = getattr(tree_model, 'shadow_tree', None)
        raw_names = getattr(shadow_tree, 'feature_names', None)

        if raw_names is None:
            return []
        try:
            names = list(raw_names)
        except TypeError:
            return []
        try:
            shadow_tree.feature_names = names
        except (AttributeError, TypeError):
            pass
        return names

    def _populate_features(self):
        _, tree_model = self._current_tree_model()
        features = self._feature_names(tree_model) if tree_model is not None else []

        self.feature_1_combo.blockSignals(True)
        self.feature_2_combo.blockSignals(True)
        self.feature_1_combo.clear()
        self.feature_2_combo.clear()
        self.feature_1_combo.addItem('None', userData=None)
        self.feature_2_combo.addItem('None', userData=None)

        for feature in features:
            self.feature_1_combo.addItem(str(feature), userData=feature)
            self.feature_2_combo.addItem(str(feature), userData=feature)

        self.feature_1_combo.setCurrentIndex(1 if features else 0)
        self.feature_2_combo.setCurrentIndex(2 if len(features) > 1 else 0)
        enabled = bool(features)
        self.feature_1_combo.setEnabled(enabled)
        self.feature_2_combo.setEnabled(enabled)
        self.feature_1_combo.blockSignals(False)
        self.feature_2_combo.blockSignals(False)

    def _mode_changed(self, index: int):
        self._update_feature_visibility()
        self._update_plot()

    def _target_changed(self, index: int):
        self._populate_features()
        self._update_plot()

    def _update_feature_visibility(self):
        visible = self.mode_tabs.currentIndex() == self.feature_space_index
        self.feature_1_label.setVisible(visible)
        self.feature_1_combo.setVisible(visible)
        self.feature_2_label.setVisible(visible)
        self.feature_2_combo.setVisible(visible)

    def _selected_features(self) -> Optional[list[Any]]:
        features = [
            feature
            for feature in (self.feature_1_combo.currentData(), self.feature_2_combo.currentData())
            if feature is not None
        ]

        if len(features) == 2 and features[0] == features[1]:
            return None

        return features

    def _update_plot(self, index: int = -1):
        _, tree_model = self._current_tree_model()

        if tree_model is None:
            self.plot.show_message('No rule-extraction results are available.')
            return

        mode = self.mode_tabs.currentIndex()
        if mode == self.tree_index:
            self.plot.show_tree(tree_model, show_progress=not self._initializing)
        elif mode == self.leaf_index:
            self.plot.show_leaf_distributions(tree_model)
        else:
            features = self._selected_features()
            if not features:
                message = (
                    'Select at least one feature.'
                    if features == []
                    else 'Select two different features.'
                )
                self.plot.show_message(message)
                return
            self.plot.show_feature_space(tree_model, features)
