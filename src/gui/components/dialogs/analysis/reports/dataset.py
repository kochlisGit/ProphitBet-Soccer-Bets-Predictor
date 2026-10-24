import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTabBar,
    QTableView,
    QVBoxLayout,
    QWidget
)


@dataclass(frozen=True)
class _Alert:
    """ One profiling alert displayed in the alerts table. """

    fragments: tuple[tuple[str, bool], ...]
    category: str

    @property
    def text(self) -> str:
        return ''.join(text for text, _ in self.fragments)


class _CountBadge(QWidget):
    """ Small badge displaying the number of alerts. """

    def __init__(self, count: int = 0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._count = count

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_count(self, count: int):
        self._count = count
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        metrics = QFontMetrics(self.font())
        width = metrics.horizontalAdvance(str(self._count)) + 18
        height = metrics.height() + 8
        return QSize(max(width, 34), max(height, 24))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor('#3949AB'))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, str(self._count))


class _ProfilingTable(QTableView):
    """ Base table used by the overview and alerts pages. """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._row_height = 52

        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QTableView.SelectionMode.NoSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(self._row_height)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)

    def fit_height_to_contents(self):
        model = self.model()
        rows = model.rowCount() if model else 0
        self.setFixedHeight(rows * self._row_height + self.frameWidth() * 2)


class _AlertsTableModel(QAbstractTableModel):
    """ Model containing profiling alerts. """

    FragmentsRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._alerts: list[_Alert] = []

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._alerts)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else 2

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        alert = self._alerts[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return alert.text if index.column() == 0 else alert.category
        if role == self.FragmentsRole and index.column() == 0:
            return alert.fragments
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        return None

    def set_alerts(self, alerts: list[_Alert]):
        self.beginResetModel()
        self._alerts = alerts
        self.endResetModel()


class _AlertTextDelegate(QStyledItemDelegate):
    """ Paint alert descriptions with highlighted variable names. """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)

        text = option.text
        option.text = ''

        option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget)
        fragments = index.data(_AlertsTableModel.FragmentsRole)
        fragments = fragments or ((text, False),)

        painter.save()

        text_rect = option.rect.adjusted(8, 0, -8, 0)
        painter.setClipRect(text_rect)
        painter.setFont(option.font)
        metrics = QFontMetrics(option.font)
        baseline = text_rect.center().y() + (metrics.ascent() - metrics.descent()) // 2

        normal = option.palette.color(QPalette.ColorRole.Text)
        highlighted = option.palette.color(QPalette.ColorRole.Link)
        x = text_rect.left()
        for fragment, is_highlighted in fragments:
            painter.setPen(highlighted if is_highlighted else normal)
            painter.drawText(x, baseline, fragment)
            x += metrics.horizontalAdvance(fragment)
        painter.restore()


class _AlertBadgeDelegate(QStyledItemDelegate):
    """ Paint alert categories as rounded badges. """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)

        option.text = ''
        option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = option.font
        font.setPointSize(font.pointSize() - 1)
        font.setBold(True)
        painter.setFont(font)

        category = str(index.data(Qt.ItemDataRole.DisplayRole))
        metrics = QFontMetrics(font)
        content_rect = option.rect.adjusted(8, 0, -8, 0)
        width = min(metrics.horizontalAdvance(category) + 20, content_rect.width())
        height = min(28, option.rect.height() - 12)
        rect = QRect(content_rect.left(), content_rect.center().y() - height // 2, width, height)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#3949AB'))
        painter.drawRoundedRect(rect, 8, 8)
        painter.setPen(option.palette.color(QPalette.ColorRole.ButtonText))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, category)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        category = str(index.data(Qt.ItemDataRole.DisplayRole))
        width = QFontMetrics(option.font).horizontalAdvance(category)
        return QSize(width + 36, 48)


class _OverviewTableModel(QAbstractTableModel):
    """Model containing dataset overview statistics."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rows: list[tuple[str, str]] = []

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else 2

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        column = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._rows[row][column]
        if role == Qt.ItemDataRole.FontRole and column == 0:
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        return None

    def set_dataframe(self, df: pd.DataFrame):
        def format_value(val: Any) -> str:
            try:
                return '—' if pd.isna(val) else str(val)
            except (TypeError, ValueError):
                return str(val)

        if df.shape[1] != 1:
            raise ValueError('Overview dataframe must contain exactly one data column.')

        self._rows = [(str(index), format_value(val=value)) for index, value in df.iloc[:, 0].items()]

        self.beginResetModel()
        self.endResetModel()


class _OverviewTextDelegate(QStyledItemDelegate):
    """ Paint overview values with an eight-pixel horizontal margin. """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)

        text = option.text
        option.text = ''
        option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget)

        painter.save()
        rect = option.rect.adjusted(8, 0, -8, 0)
        painter.setClipRect(rect)
        painter.setFont(option.font)
        painter.setPen(option.palette.color(QPalette.ColorRole.Text))
        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        painter.restore()


class _ProfilingPage(QWidget):
    """Base page containing a vertically scrollable table."""

    def __init__(
            self,
            table: _ProfilingTable,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.table = table

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)
        scroll_layout.addWidget(table)
        scroll_layout.addStretch()

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidget(scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setMinimumHeight(250)
        self.scroll_area.setMaximumHeight(600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)
        layout.addWidget(self.scroll_area)


class _AlertsPage(_ProfilingPage):
    """Dataset profiling alerts page."""

    def __init__(self, alerts: list[_Alert], parent: Optional[QWidget] = None):
        self.model = _AlertsTableModel()
        self.model.set_alerts(alerts)

        table = _ProfilingTable()
        table.setModel(self.model)
        table.setItemDelegateForColumn(0, _AlertTextDelegate(table))
        table.setItemDelegateForColumn(1, _AlertBadgeDelegate(table))

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        table.fit_height_to_contents()

        super().__init__(table, parent=parent)

        self.model.setParent(self)
        self.table.setParent(self.scroll_area.widget())

    def set_alerts(self, alerts: list[_Alert]):
        self.model.set_alerts(alerts)
        self.table.fit_height_to_contents()


class _OverviewPage(_ProfilingPage):
    """Dataset statistics page."""

    def __init__(self, df: pd.DataFrame, parent: Optional[QWidget] = None):
        self.model = _OverviewTableModel()
        self.model.set_dataframe(df)

        table = _ProfilingTable()
        table.setModel(self.model)
        table.setItemDelegate(_OverviewTextDelegate(table))

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)

        table.setColumnWidth(1, 240)
        table.fit_height_to_contents()

        super().__init__(table, parent=parent)

        self.model.setParent(self)
        self.table.setParent(self.scroll_area.widget())

    def set_dataframe(self, dataframe: pd.DataFrame):
        self.model.set_dataframe(dataframe)
        self.table.fit_height_to_contents()


class DatasetOverviewReport(QFrame):
    """ Compact dataset-profiling widget inspired by data-profiling reports.

        Attributes
        ----------
        :param overview: (pd.DataFrame) One-column dataframe. Its index contains statistic names and
                                        its single column contains the corresponding values.
        :param imbalanced_variables: (List[Tuple[str, float]] | None) imbalanced variables and their
                                     normalized entropy scores.
        :param issues: (Dict[str, Any]) dict that maps an issue name to its description and the affected variables
                       paired with the metric value that triggered the issue.
        :param correlated_pairs: (List[Tuple[str, str, float]] | None) highly correlated variable pairs
                                 and their correlation values.
        :param parent: (QWidget | None) parent widget.
    """

    def __init__(
            self,
            overview: pd.DataFrame,
            imbalanced_variables: Optional[List[Tuple[str, float]]] = None,
            issues: Optional[Dict[str, Tuple[str, List[Tuple[str, float]]]]] = None,
            correlated_pairs: Optional[List[Tuple[str, str, float]]] = None,
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._configure_widget()
        alerts = []

        if correlated_pairs is not None:
            for var1, var2, correlation in correlated_pairs:
                alerts.append(
                    _Alert(
                        fragments=(
                            (var1, True),
                            (' is highly correlated with ', False),
                            (var2, True),
                            (f' ({self._format_metric(correlation)}). One of the two variables might be redundant.', False),
                        ),
                        category='High Correlation'
                    )
                )

        if imbalanced_variables is not None:
            for variable, entropy in imbalanced_variables:
                alerts.append(
                    _Alert(
                        fragments=(
                            (variable, True),
                            (
                                f' is highly imbalanced (Norm. Entropy: {self._format_metric(entropy)}). '
                                'Low frequency values might be ignored.', False
                            ),
                        ),
                        category='Imbalance'
                    )
                )

        if issues is not None:
            for issue, (description, columns) in issues.items():
                for column, value in columns:
                    alerts.append(
                        _Alert(
                            fragments=(
                                (column, True),
                                (f' {description} ({value})', False),
                            ),
                            category=issue
                        )
                    )

        self.tabs = QTabBar(self)
        self._configure_tabs(len(alerts))

        self.overview_page = _OverviewPage(overview, self)
        self.alerts_page = _AlertsPage(alerts, self)
        self.alerts_page.hide()

        title = QLabel('Dataset Description', self)
        title_font = title.font()
        title_font.setBold(True)
        if title_font.pointSize() > 0:
            title_font.setPointSize(title_font.pointSize() + 2)
        elif title_font.pixelSize() > 0:
            title_font.setPixelSize(title_font.pixelSize() + 2)
        title.setFont(title_font)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 20)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(self.tabs)
        layout.addWidget(self.overview_page)
        layout.addWidget(self.alerts_page)

        self.tabs.currentChanged.connect(self._change_page)

    @staticmethod
    def _format_metric(value: float) -> str:
        """Format analysis values compactly without hiding the analyzer's precision."""

        return f'{float(value):.3f}'.rstrip('0').rstrip('.')

    @classmethod
    def _issue_metric(cls, issue: str, value: float) -> str:
        if issue == 'High Missing':
            return f'missing: {cls._format_metric(value)}%'

        labels = {
            'Low Variance': 'spread',
            'Excess Skewness': 'absolute skewness',
            'Excess Kurtosis': 'kurtosis',
            'Excess Outliers': 'normalized mean-median difference',
        }
        return f'{labels.get(issue, "value")}: {cls._format_metric(value)}'

    def _configure_widget(self):
        font = self.font()

        if font.pointSize() > 0:
            font.setPointSize(font.pointSize() + 2)
        elif font.pixelSize() > 0:
            font.setPixelSize(font.pixelSize() + 2)

        self.setFont(font)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(900, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

    def _configure_tabs(self, alert_count: int):
        font = self.tabs.font()
        font.setBold(True)

        if font.pointSize() > 0:
            font.setPointSize(font.pointSize() + 1)
        else:
            font.setPixelSize(font.pixelSize() + 1)

        self.tabs.setFont(font)
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)

        self.overview_index = self.tabs.addTab('Overview')
        self.alerts_index = self.tabs.addTab('Alerts')

        spacing = QWidget(self.tabs)
        spacing.setFixedWidth(8)
        spacing.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.tabs.setTabButton(self.overview_index, QTabBar.ButtonPosition.RightSide, spacing)

        self.alert_count = _CountBadge(alert_count, self.tabs)
        self.tabs.setTabButton(self.alerts_index, QTabBar.ButtonPosition.RightSide, self.alert_count)
        self.tabs.setFixedWidth(self.tabs.sizeHint().width() + 8)

    def _change_page(self, index: int):
        show_overview = index == self.overview_index
        self.overview_page.setVisible(show_overview)
        self.alerts_page.setVisible(not show_overview)
