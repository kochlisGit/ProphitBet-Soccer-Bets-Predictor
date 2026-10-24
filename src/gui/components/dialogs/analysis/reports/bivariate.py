import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Any, Dict, FrozenSet, List, Iterable, Optional, Set
from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, Signal, Qt
from PySide6.QtGui import QColor, QCursor, QFont, QPalette, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableView,
    QToolTip,
    QVBoxLayout,
    QWidget
)


def _increase_font(font: QFont, amount: int = 2) -> QFont:
    """ Return a copy of a font increased by the requested amount. """

    font = QFont(font)
    if font.pointSize() > 0:
        font.setPointSize(font.pointSize() + amount)
    elif font.pixelSize() > 0:
        font.setPixelSize(font.pixelSize() + amount)
    return font


def _format_number(value: Any) -> str:
    """ Format a numeric report value compactly. """

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
        return f'{number:,.2f}'.rstrip('0').rstrip('.')
    return str(value)


class _CorrelationTableModel(QAbstractTableModel):
    """ Read-only model for one feature-correlation matrix. """

    def __init__(
            self,
            correlated_pairs: Optional[Set[FrozenSet[Any]]] = None,
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._dataframe = pd.DataFrame()
        self._diag_mask: Optional[np.ndarray] = None
        self._correlated_pairs = correlated_pairs or set()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else self._dataframe.shape[0]

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else self._dataframe.shape[1]

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        column = index.column()
        
        if self._is_masked(row, column):
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
                return None
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignCenter
            return None

        value = self._dataframe.iat[row, column]
        variables = frozenset((self._dataframe.index[row], self._dataframe.columns[column]))
        
        if role == Qt.ItemDataRole.DisplayRole:
            return _format_number(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        if variables in self._correlated_pairs:
            if role == Qt.ItemDataRole.BackgroundRole:
                return QColor('#3949AB')
            if role == Qt.ItemDataRole.ForegroundRole:
                return QColor('white')
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal:
            if section >= self._dataframe.shape[1]:
                return None
            variable = self._dataframe.columns[section]
        else:
            if section >= self._dataframe.shape[0]:
                return None
            variable = self._dataframe.index[section]

        if role == Qt.ItemDataRole.DisplayRole:
            return str(variable)
        return None

    def _is_masked(self, row: int, column: int) -> bool:
        if self._diag_mask is None:
            return False
        if self._diag_mask.shape != self._dataframe.shape:
            return False
        
        return bool(self._diag_mask[row, column])

    def set_dataframe(self, dataframe: Optional[pd.DataFrame], diag_mask: Optional[np.ndarray] = None):
        self.beginResetModel()
        self._dataframe = pd.DataFrame() if dataframe is None else dataframe
        self._diag_mask = diag_mask
        self.endResetModel()


class _CheckableComboBox(QComboBox):
    """ Compact multi-selection combobox with checkable popup rows. """

    selectionChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setEditable(True)

        line_edit = self.lineEdit()
        line_edit.setReadOnly(True)
        line_edit.setPlaceholderText('Select columns')
        self.setModel(QStandardItemModel(self))
        self.view().viewport().installEventFilter(self)

    def set_items(self, values: Iterable[Any], checked_count: int = 10):
        model = self.model()
        model.clear()

        for index, value in enumerate(values):
            item = QStandardItem(str(value))
            item.setData(value, Qt.ItemDataRole.UserRole)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            state = Qt.CheckState.Checked if index < checked_count else Qt.CheckState.Unchecked
            item.setData(state, Qt.ItemDataRole.CheckStateRole)
            model.appendRow(item)
        self._update_summary()

    def checked_values(self) -> List[Any]:
        values = []
        model = self.model()
        for row in range(model.rowCount()):
            item = model.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                values.append(item.data(Qt.ItemDataRole.UserRole))
        return values

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.view().viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            index = self.view().indexAt(event.position().toPoint())
            if index.isValid():
                item = self.model().itemFromIndex(index)
                checked = item.checkState() == Qt.CheckState.Checked
                item.setCheckState(
                    Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked
                )
                self._update_summary()
                self.selectionChanged.emit()
                return True
        return super().eventFilter(watched, event)

    def _update_summary(self):
        count = len(self.checked_values())
        self.setEditText(f'{count} column' if count == 1 else f'{count} columns')


class _CorrelationHeatmap(QWidget):
    """ Seaborn heatmap that retains only its current rendered plot. """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_key: Optional[tuple[Any, ...]] = None
        self._canvas: Optional[FigureCanvas] = None
        self._axis = None

        self._stack = QStackedWidget(self)
        self._empty_label = QLabel('Select columns to display a heatmap.', self)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(f'background-color: #CFD8DC;')
        self._stack.addWidget(self._empty_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._stack)

    def set_heatmap(
            self,
            correlation: str,
            dataframe: Optional[pd.DataFrame],
            columns: list[Any],
            theme: str,
            diag_mask: Optional[np.ndarray]
    ):
        if dataframe is None or not columns:
            self._stack.setCurrentWidget(self._empty_label)
            return

        columns = [column for column in columns if column in dataframe.columns]
        if not columns:
            self._stack.setCurrentWidget(self._empty_label)
            return

        masked = diag_mask is not None
        key = (correlation, tuple(columns), masked)
        colormap = sns.color_palette(theme, as_cmap=True)
        if key != self._current_key or self._canvas is None:
            self._create_heatmap(
                key,
                dataframe.loc[columns, columns],
                colormap,
                diag_mask
            )
        elif self._axis is not None and self._axis.collections:
            self._axis.collections[0].set_cmap(colormap)
            self._canvas.draw_idle()

        self._stack.setCurrentWidget(self._canvas)

    def _create_heatmap(
            self,
            key: tuple[Any, ...],
            dataframe: pd.DataFrame,
            colormap,
            diag_mask: Optional[np.ndarray]
    ):
        self._clear_heatmap()
        figure = Figure(figsize=(5, 4), dpi=100, facecolor='#CFD8DC')
        axis = figure.add_subplot(111)
        axis.set_facecolor('#CFD8DC')
        sns.heatmap(
            dataframe,
            mask=diag_mask,
            cmap=colormap,
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt='.2f',
            square=False,
            linewidths=0.5,
            ax=axis
        )
        axis.tick_params(axis='x', rotation=45, labelsize=9)
        axis.tick_params(axis='y', rotation=0, labelsize=9)
        figure.tight_layout(pad=1.0)

        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas.setStyleSheet(f'background-color: #CFD8DC;')
        self._stack.addWidget(canvas)
        self._current_key = key
        self._canvas = canvas
        self._axis = axis
        canvas.draw()

    def _clear_heatmap(self):
        if self._canvas is None:
            return
        self._stack.removeWidget(self._canvas)
        self._canvas.figure.clear()
        self._canvas.deleteLater()
        self._canvas = None
        self._axis = None
        self._current_key = None


class _InteractionPlot(QWidget):
    """ Fixed-size seaborn scatter plot retaining only the current figure. """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.PLOT_WIDTH = 460
        self.PLOT_HEIGHT = 370
        self.PLOT_DPI = 100

        self.setFixedSize(self.PLOT_WIDTH, self.PLOT_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._canvas: Optional[FigureCanvas] = None
        self._hover_index: Optional[int] = None
        self._stack = QStackedWidget(self)
        self._stack.setFixedSize(self.PLOT_WIDTH, self.PLOT_HEIGHT)
        self._empty_label = QLabel(self)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setFixedSize(self.PLOT_WIDTH, self.PLOT_HEIGHT)
        self._empty_label.setStyleSheet(f'background-color: #CFD8DC; padding: 20px;')
        self._stack.addWidget(self._empty_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._stack)

        self._draw_empty('Select two variables to view their interaction.')

    def _draw_empty(self, message: str):
        self._clear_scatter()
        self._empty_label.setText(message)
        self._stack.setCurrentWidget(self._empty_label)

    def set_interaction(self, x_variable: Any, y_variable: Any, dataframe: Optional[pd.DataFrame]):
        if dataframe is None or x_variable == y_variable:
            message = f'No interaction is available for {x_variable} and {y_variable}.'
            self._draw_empty(message)
            return

        canvas = self._create_scatter(x_variable=x_variable, y_variable=y_variable, dataframe=dataframe)

        if canvas is None:
            message = f'No valid observations are available for {x_variable} and {y_variable}.'
            self._draw_empty(message)
            return

        self._stack.setCurrentWidget(canvas)

    def _create_scatter(self, x_variable: Any, y_variable: Any, dataframe: pd.DataFrame) -> Optional[FigureCanvas]:
        values = dataframe.loc[:, [x_variable, y_variable]]
        values = values.apply(pd.to_numeric, errors='coerce').dropna()

        if values.empty:
            return None

        self._clear_scatter()
        width = self.PLOT_WIDTH / self.PLOT_DPI
        height = self.PLOT_HEIGHT / self.PLOT_DPI
        figure = Figure(figsize=(width, height), dpi=self.PLOT_DPI, facecolor='#CFD8DC')
        axis = figure.add_subplot(111)
        axis.set_facecolor('#CFD8DC')

        sns.scatterplot(
            data=values,
            x=x_variable,
            y=y_variable,
            color=self.palette().color(QPalette.ColorRole.Highlight).name(),
            edgecolor='none',
            s=48,
            ax=axis
        )
        axis.set_title(f'{x_variable} - {y_variable}', fontsize=12)
        axis.set_xlabel(str(x_variable), fontsize=11, fontweight='bold')
        axis.set_ylabel(str(y_variable), fontsize=11, fontweight='bold')
        axis.tick_params(axis='both', labelsize=10)
        axis.grid(True, alpha=0.3)
        figure.tight_layout(pad=1.0)

        canvas = FigureCanvas(figure)
        canvas.setFixedSize(self.PLOT_WIDTH, self.PLOT_HEIGHT)
        canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        canvas.setStyleSheet(f'background-color: #CFD8DC;')
        self._stack.addWidget(canvas)

        key = (x_variable, y_variable)
        self._canvas = canvas
        self._hover_index = None

        collection = axis.collections[-1]
        canvas.mpl_connect(
            'motion_notify_event',
            lambda event, plot_axis=axis, pair=key, points=values,
                   scatter=collection: self._show_hover_message(event, plot_axis, pair, points, scatter)
        )
        canvas.draw()
        return canvas

    def _show_hover_message(self, event, axis, pair, values, scatter):
        if event.inaxes is not axis:
            if self._hover_index is not None:
                self._hover_index = None
                QToolTip.hideText()
            return

        contains, info = scatter.contains(event)
        indices = info.get('ind', [])

        if not contains or len(indices) == 0:
            if self._hover_index is not None:
                self._hover_index = None
                QToolTip.hideText()
            return

        index = int(indices[0])

        if self._hover_index == index:
            return

        self._hover_index = index
        x_variable, y_variable = pair
        x_value = values.iloc[index, 0]
        y_value = values.iloc[index, 1]

        message = f'({x_variable}, {y_variable}) = ({_format_number(x_value)}, {_format_number(y_value)})'
        QToolTip.showText(QCursor.pos(), message, self)

    def _clear_scatter(self):
        if self._canvas is None:
            return

        self._stack.removeWidget(self._canvas)
        self._canvas.figure.clear()
        self._canvas.deleteLater()
        self._canvas = None
        self._hover_index = None


class BivariateReport(QFrame):
    """ Two-panel report for feature interactions and correlation matrices.

        Attributes
        ----------
        :param feature_correlations: Mapping returned under
            ``Bivariate['Feature Correlations']``.
        :param interactions: Mapping returned under ``Bivariate['Interactions']``.
        :param parent: (QWidget | None) parent widget.
        :param correlated_pairs: Pairs whose cells should be highlighted in the
            correlation table.
    """

    _correlation_methods = ('Pearson', 'Spearman', 'Kendall', 'Distance')

    def __init__(
            self,
            feature_correlations: Dict[str, pd.DataFrame],
            interactions: Dict[str, pd.DataFrame],
            parent: Optional[QWidget] = None,
            correlated_pairs: Optional[Iterable[tuple[Any, Any]]] = None
    ):
        super().__init__(parent)

        self._heatmap_themes = (
            'vlag_r', 'mako_r', 'flare_r', 'crest', 'coolwarm', 'Spectral', 'RdBu', 'seismic', 'Blues', 'Greens',
            'Purples', 'GnBu', 'PuBu', 'viridis_r', 'plasma_r'
        )

        self._feature_correlations = feature_correlations
        self._interactions = interactions
        self._changing_variables = False
        self._heatmap_column_names = ()
        self._correlated_pairs = self._extract_correlated_pairs(correlated_pairs=correlated_pairs)
        self._diag_mask = self._precompute_diag_mask()

        self._configure_widget()
        self._populate_variables()
        self._update_correlation()

    @property
    def x_variable(self):
        index = self.x_variable_combo.currentIndex()
        return None if index < 0 else self.x_variable_combo.itemData(index)

    @property
    def y_variable(self):
        index = self.y_variable_combo.currentIndex()
        return None if index < 0 else self.y_variable_combo.itemData(index)

    @staticmethod
    def _extract_correlated_pairs(correlated_pairs: Optional[Iterable[tuple[Any, Any]]]) -> Set[FrozenSet[Any]]:
        if correlated_pairs is None:
            return set()

        pairs = set()
        for pair in correlated_pairs:
            if len(pair) < 2:
                continue
            pairs.add(frozenset((pair[0], pair[1])))
        return pairs

    def _precompute_diag_mask(self) -> Optional[np.ndarray]:
        for dataframe in self._feature_correlations.values():
            rows, columns = dataframe.shape
            if rows == columns and rows > 0:
                return np.triu(np.ones((rows, columns), dtype=bool), k=1)
        return None

    def _configure_widget(self):
        self.setFont(_increase_font(self.font()))
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(900, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.interactions_panel = QFrame(self)
        self.interactions_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.correlations_panel = QFrame(self)
        self.correlations_panel.setFrameShape(QFrame.Shape.StyledPanel)

        self._configure_interactions_panel()
        self._configure_correlations_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setOpaqueResize(False)
        splitter.addWidget(self.interactions_panel)
        splitter.addWidget(self.correlations_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 500])

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)
        layout.addWidget(splitter)

    @staticmethod
    def _title_label(text: str, parent: QWidget) -> QLabel:
        label = QLabel(text, parent)
        font = label.font()
        font.setBold(True)
        label.setFont(_increase_font(font))
        return label

    def _configure_interactions_panel(self):
        title = self._title_label('Interactions', self.interactions_panel)
        self.x_variable_combo = QComboBox(self.interactions_panel)
        self.y_variable_combo = QComboBox(self.interactions_panel)
        self.x_variable_combo.setFixedWidth(100)
        self.y_variable_combo.setFixedWidth(100)
        self.x_variable_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.y_variable_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        selectors_layout = QHBoxLayout()
        selectors_layout.setContentsMargins(0, 0, 0, 0)
        selectors_layout.setSpacing(14)

        x_layout = QHBoxLayout()
        x_layout.setContentsMargins(0, 0, 0, 0)
        x_layout.setSpacing(6)
        x_layout.addWidget(QLabel('X-Axis', self.interactions_panel))
        x_layout.addWidget(self.x_variable_combo, 1)

        y_layout = QHBoxLayout()
        y_layout.setContentsMargins(0, 0, 0, 0)
        y_layout.setSpacing(6)
        y_layout.addWidget(QLabel('Y-Axis', self.interactions_panel))
        y_layout.addWidget(self.y_variable_combo, 1)

        selectors_layout.addStretch(1)
        selectors_layout.addLayout(x_layout)
        selectors_layout.addLayout(y_layout)
        selectors_layout.addStretch(1)

        self.interaction_plot = _InteractionPlot(self.interactions_panel)

        layout = QVBoxLayout(self.interactions_panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addLayout(selectors_layout)
        layout.addWidget(self.interaction_plot, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        layout.addStretch(1)

        self.x_variable_combo.currentIndexChanged.connect(self._x_variable_changed)
        self.y_variable_combo.currentIndexChanged.connect(self._y_variable_changed)

    def _configure_correlations_panel(self):
        title = self._title_label('Correlations', self.correlations_panel)

        correlation_label = QLabel('Type:', self.correlations_panel)
        self.correlation_combo = QComboBox(self.correlations_panel)
        self.correlation_combo.addItems(self._correlation_methods)
        self.correlation_combo.setCurrentText('Pearson')
        self.diag_mask_checkbox = QCheckBox('Diag. Mask', self.correlations_panel)
        self.diag_mask_checkbox.setChecked(True)
        self.show_heatmap_button = QPushButton('Show Heatmap', self.correlations_panel)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        toolbar_layout.addWidget(self.show_heatmap_button)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(correlation_label)
        toolbar_layout.addWidget(self.correlation_combo)
        toolbar_layout.addWidget(self.diag_mask_checkbox)

        self.correlation_model = _CorrelationTableModel(correlated_pairs=self._correlated_pairs, parent=self)
        self.correlation_table = QTableView(self.correlations_panel)
        self.correlation_table.setModel(self.correlation_model)
        self.correlation_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.correlation_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.correlation_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.correlation_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.correlation_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.correlation_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.correlation_table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.correlation_table.setWordWrap(False)
        self.correlation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.correlation_table.horizontalHeader().setDefaultSectionSize(90)
        self.correlation_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.correlation_table.verticalHeader().setDefaultSectionSize(34)

        self._configure_heatmap_dialog()

        layout = QVBoxLayout(self.correlations_panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addLayout(toolbar_layout)
        layout.addWidget(self.correlation_table, 1)

        self.correlation_combo.currentTextChanged.connect(self._update_correlation)
        self.diag_mask_checkbox.toggled.connect(lambda checked: self._update_correlation())
        self.show_heatmap_button.clicked.connect(self._show_heatmap)

    def _configure_heatmap_dialog(self):
        self.heatmap_dialog = QDialog(self)
        self.heatmap_dialog.setWindowTitle('Feature Correlation Heatmap')
        self.heatmap_dialog.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self.heatmap_dialog.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.heatmap_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.heatmap_dialog.setModal(False)
        self.heatmap_dialog.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.heatmap_dialog.resize(900, 520)

        heatmap_controls = QHBoxLayout()
        heatmap_controls.setContentsMargins(12, 12, 12, 0)
        heatmap_controls.setSpacing(8)

        heatmap_theme_label = QLabel('Theme:', self.heatmap_dialog)
        self.heatmap_theme_combo = QComboBox(self.heatmap_dialog)
        self.heatmap_theme_combo.addItems(self._heatmap_themes)
        self.heatmap_theme_combo.setCurrentIndex(0)

        heatmap_columns_label = QLabel('Columns:', self.heatmap_dialog)
        self.heatmap_columns_combo = _CheckableComboBox(self.heatmap_dialog)
        self.heatmap_columns_combo.setMinimumWidth(150)

        heatmap_controls.addWidget(heatmap_theme_label)
        heatmap_controls.addWidget(self.heatmap_theme_combo)
        heatmap_controls.addWidget(heatmap_columns_label)
        heatmap_controls.addWidget(self.heatmap_columns_combo, 1)

        self.correlation_heatmap = _CorrelationHeatmap(self.heatmap_dialog)
        heatmap_layout = QVBoxLayout(self.heatmap_dialog)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_layout.setSpacing(8)
        heatmap_layout.addLayout(heatmap_controls)
        heatmap_layout.addWidget(self.correlation_heatmap, 1)
        self.heatmap_theme_combo.currentTextChanged.connect(lambda theme: self._update_heatmap())
        self.heatmap_columns_combo.selectionChanged.connect(self._update_heatmap)

    def _variables(self) -> List[Any]:
        variables = []
        for dataframe in self._feature_correlations.values():
            for variable in dataframe.columns:
                if variable not in variables:
                    variables.append(variable)

        for dataframe in self._interactions.values():
            for variable in dataframe.columns:
                if variable not in variables:
                    variables.append(variable)
        return variables

    def _populate_variables(self):
        variables = self._variables()
        self.x_variable_combo.blockSignals(True)
        self.y_variable_combo.blockSignals(True)
        self.x_variable_combo.clear()
        self.y_variable_combo.clear()

        for variable in variables:
            self.x_variable_combo.addItem(str(variable), userData=variable)
            self.y_variable_combo.addItem(str(variable), userData=variable)

        self.x_variable_combo.setCurrentIndex(0 if variables else -1)
        y_index = 1 if len(variables) > 1 else (0 if variables else -1)
        self.y_variable_combo.setCurrentIndex(y_index)
        self.x_variable_combo.setEnabled(bool(variables))
        self.y_variable_combo.setEnabled(bool(variables))
        self.x_variable_combo.blockSignals(False)
        self.y_variable_combo.blockSignals(False)
        self._update_interaction()

    def _x_variable_changed(self, index: int):
        if self._changing_variables:
            return
        self._ensure_distinct_variable(changed_x=True)
        self._update_interaction()

    def _y_variable_changed(self, index: int):
        if self._changing_variables:
            return
        self._ensure_distinct_variable(changed_x=False)
        self._update_interaction()

    def _ensure_distinct_variable(self, changed_x: bool):
        if self.x_variable != self.y_variable:
            return
        if self.x_variable_combo.count() < 2:
            return

        self._changing_variables = True
        source = self.x_variable_combo if changed_x else self.y_variable_combo
        target = self.y_variable_combo if changed_x else self.x_variable_combo
        target.setCurrentIndex((source.currentIndex() + 1) % source.count())
        self._changing_variables = False

    def _interaction_dataframe(self, x_variable: Any, y_variable: Any) -> Optional[pd.DataFrame]:
        for candidate in self._interactions.values():
            if x_variable in candidate.columns and y_variable in candidate.columns:
                dataframe = candidate
                break

        return dataframe

    def _update_interaction(self):
        x_variable = self.x_variable
        y_variable = self.y_variable
        if x_variable is None or y_variable is None:
            self.interaction_plot._draw_empty('No variables are available for interaction analysis.')
            return

        self.interaction_plot.set_interaction(
            x_variable=x_variable,
            y_variable=y_variable,
            dataframe=self._interaction_dataframe(x_variable, y_variable)
        )

    def _mask_for_dataframe(self, dataframe: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        if not self.diag_mask_checkbox.isChecked():
            return None
        if dataframe is None or self._diag_mask is None:
            return None
        if dataframe.shape != self._diag_mask.shape:
            return None
        return self._diag_mask

    def _update_correlation(self, correlation: Optional[str] = None):
        correlation = correlation or self.correlation_combo.currentText()
        dataframe = self._feature_correlations.get(correlation)
        self._populate_heatmap_columns(dataframe)
        self.correlation_model.set_dataframe(dataframe=dataframe, diag_mask=self._mask_for_dataframe(dataframe))
        if self.heatmap_dialog.isVisible():
            self._update_heatmap()

    def _populate_heatmap_columns(self, dataframe: Optional[pd.DataFrame]):
        columns = () if dataframe is None else tuple(dataframe.columns)

        if columns == self._heatmap_column_names:
            return

        self._heatmap_column_names = columns
        self.heatmap_columns_combo.set_items(columns, checked_count=15)

    def _heatmap_mask(
            self,
            dataframe: Optional[pd.DataFrame],
            columns: list[Any]
    ) -> Optional[np.ndarray]:
        full_mask = self._mask_for_dataframe(dataframe)

        if dataframe is None or full_mask is None or not columns:
            return None

        positions = [dataframe.columns.get_loc(column) for column in columns]
        return full_mask[np.ix_(positions, positions)]

    def _update_heatmap(self):
        correlation = self.correlation_combo.currentText()
        dataframe = self._feature_correlations.get(correlation)
        columns = self.heatmap_columns_combo.checked_values()
        self.correlation_heatmap.set_heatmap(
            correlation=correlation,
            dataframe=dataframe,
            columns=columns,
            theme=self.heatmap_theme_combo.currentText(),
            diag_mask=self._heatmap_mask(dataframe, columns)
        )

    def _show_heatmap(self):
        self._update_heatmap()
        self.heatmap_dialog.showMaximized()

    def closeEvent(self, event):  # noqa: N802
        self.heatmap_dialog.close()
        super().closeEvent(event)
