import re
import pandas as pd
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QItemSelection,
    QItemSelectionModel,
    QRect,
    QSortFilterProxyModel,
    QStandardPaths,
    Qt,
    QTimer
)
from PySide6.QtGui import QAction, QFont, QKeyEvent, QKeySequence, QMouseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget
)
from src.core.io.exports import export_table


class DataFrameTableModel(QAbstractTableModel):
    """ Fast DataFrame-backed table model for use with QTableView.

        Attributes
        ----------
        :param df: (pd.DataFrame) dataframe to be displayed.
        :param readonly: (bool) whether table cells are read-only and cannot be changed.
        :param parent: (QWidget | None) parent of the table.
    """

    def __init__(self, df: pd.DataFrame, readonly: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.columns = df.columns.tolist()
        self._column_labels = self.columns.copy()

        self._df = df.copy()
        self._readonly = readonly
        self._missing_rows = self._calculate_missing_rows()
        self._num_missing = len(self._missing_rows)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @property
    def column_labels(self) -> List[str]:
        return self._column_labels.copy()

    @property
    def missing_rows(self) -> set[int]:
        return self._missing_rows

    @property
    def num_missing(self) -> int:
        return self._num_missing

    def _calculate_missing_rows(self) -> set[int]:
        if self._df.empty:
            return set()

        mask = self._df.isna().any(axis=1).to_numpy()
        missing_selection = set(mask.nonzero()[0].tolist())
        return missing_selection

    def _refresh_missing_row(self, row: int) -> None:
        if self._df.iloc[row].isna().any():
            self._missing_rows.add(row)
        else:
            self._missing_rows.discard(row)

        self._num_missing = len(self._missing_rows)

    @lru_cache(maxsize=100_000)
    def _display_value(self, row: int, column: int) -> str:
        value = self._df.iat[row, column]
        return '' if pd.isna(value) else str(value)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0

        return self._df.shape[0]

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0

        return self._df.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        column = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_value(row, column)
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        elif role == Qt.ItemDataRole.UserRole:
            value = self._df.iat[row, column]
            return None if pd.isna(value) else value
        else:
            return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if 0 <= section < len(self._column_labels):
                    return str(self._column_labels[section])
                return None
            return str(section + 1)
        elif role == Qt.ItemDataRole.FontRole:
            font = QFont()
            font.setBold(True)
            return font

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        else:
            return None

    def set_column_labels(self, labels: Mapping[str, str]) -> None:
        """Set display-only labels for logical dataframe columns.

        Every call replaces the current aliases. Columns omitted from ``labels``
        use their logical dataframe names.
        """

        unknown_columns = labels.keys() - self.columns
        if unknown_columns:
            unknown = ', '.join(sorted(map(str, unknown_columns)))
            raise KeyError(f'Unknown columns: {unknown}.')

        column_labels = [labels.get(column, column) for column in self.columns]
        changed_columns = [
            index
            for index, (old_label, new_label) in enumerate(zip(self._column_labels, column_labels))
            if old_label != new_label
        ]
        if not changed_columns:
            return

        self._column_labels = column_labels
        self.headerDataChanged.emit(
            Qt.Orientation.Horizontal,
            min(changed_columns),
            max(changed_columns)
        )

    def replace_dataframe(self, df: pd.DataFrame) -> None:
        """Replace all rows while preserving the table's column contract."""

        columns = df.columns.tolist()
        if columns != self.columns:
            raise ValueError(f'Expected columns {self.columns}, got {columns}.')

        self.beginResetModel()
        self._df = df.copy()
        self._display_value.cache_clear()
        self._missing_rows = self._calculate_missing_rows()
        self._num_missing = len(self._missing_rows)
        self.endResetModel()

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        flags = (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

        if not self._readonly:
            flags |= Qt.ItemFlag.ItemIsEditable

        return flags

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> Optional[bool]:
        if self._readonly or not index.isValid():
            return False
        elif role not in (Qt.ItemDataRole.EditRole, Qt.ItemDataRole.DisplayRole):
            return False
        else:
            row = index.row()
            column = index.column()

            # Validating dtype if column. Incosistent dtypes will be rejected.
            old_value = self._df.iat[row, column]

            if pd.api.types.is_float_dtype(old_value):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return False
            elif pd.api.types.is_integer_dtype(old_value):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    return False
            if value == old_value:
                return False

            self._df.iat[row, column] = value
            self._display_value.cache_clear()
            self._refresh_missing_row(row)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.UserRole])
            return True

    def clear_cells(self, indexes: Iterable[QModelIndex]):
        """Clear a collection of source-model indexes efficiently."""
        if self._readonly:
            return

        coordinates = {(index.row(), index.column()) for index in indexes}

        if not coordinates:
            return

        rows_changed = set()
        for row, column in coordinates:
            self._df.iat[row, column] = ''
            rows_changed.add(row)

        self._display_value.cache_clear()
        for row in rows_changed:
            self._refresh_missing_row(row)

        rows = [row for row, _ in coordinates]
        columns = [column for _, column in coordinates]
        top_left = self.index(min(rows), min(columns))
        bottom_right = self.index(max(rows), max(columns))
        self.dataChanged.emit(
            top_left,
            bottom_right,
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.UserRole],
        )

    def modify_columns(
            self,
            columns: Sequence[str],
            data: Sequence[Sequence[object]],
            rows: Optional[Sequence[int]] = None
    ):
        """ Updates one or more columns without creating per-cell Qt items. """
        num_data_rows = len(data)

        if num_data_rows > self.rowCount():
            raise ValueError(f'Expected maximum {self.rowCount()} rows, got {num_data_rows}.')
        if rows is not None and num_data_rows > len(rows):
            raise ValueError(f'Expected maximum {len(rows)} rows, got {num_data_rows}.')
        if not columns or not data:
            return

        column_ids = [self.columns.index(column) for column in columns]
        changed_rows = []

        for i, values in enumerate(data):
            if len(values) != len(column_ids):
                raise ValueError(
                    f'Expected {len(column_ids)} values in row {i}, '
                    f'got {len(values)}.'
                )

            row_id = i if rows is None else rows[i]
            if not 0 <= row_id < self.rowCount():
                raise IndexError(f'Row index out of range: {row_id}.')

            for column_id, value in zip(column_ids, values):
                self._df.iat[row_id, column_id] = value
            changed_rows.append(row_id)

        self._display_value.cache_clear()
        for row in set(changed_rows):
            self._refresh_missing_row(row)

        top = min(changed_rows)
        bottom = max(changed_rows)
        left = min(column_ids)
        right = max(column_ids)
        self.dataChanged.emit(
            self.index(top, left),
            self.index(bottom, right),
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.UserRole]
        )

    def modify_rows(self, row_ids: List[int], items: List[List[Any]]):
        if len(row_ids) != len(items):
            raise ValueError(
                f'Expected one item row per row id, got {len(items)} item rows '
                f'for {len(row_ids)} row ids.'
            )
        if not row_ids:
            return

        # Update DataFrame in one operation.
        self._df.iloc[row_ids] = items
        self._display_value.cache_clear()
        for row_id in set(row_ids):
            self._refresh_missing_row(row_id)

        # Emit only actually affected contiguous ranges.
        sorted_rows = sorted(set(row_ids))
        column_count = self.columnCount()

        start = previous = sorted_rows[0]
        for row in sorted_rows[1:]:
            if row == previous + 1:
                previous = row
                continue

            self.dataChanged.emit(
                self.index(start, 0),
                self.index(previous, column_count - 1),
                [
                    Qt.ItemDataRole.DisplayRole,
                    Qt.ItemDataRole.UserRole
                ]
            )
            start = previous = row

        self.dataChanged.emit(
            self.index(start, 0),
            self.index(previous, column_count - 1),
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.UserRole
            ]
        )

    def find_matches(
            self,
            query: str,
            exact_match: bool,
            source_column: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """ Find source-model coordinates using vectorized pandas string operations. """
        if not query:
            return []

        query_folded = query.casefold()
        columns = (
            range(self.columnCount())
            if source_column is None
            else (source_column,)
        )
        matches: List[Tuple[int, int]] = []

        for column in columns:
            series = self._df.iloc[:, column]
            text = series.astype('string').fillna('').str.casefold()
            if exact_match:
                mask = text.eq(query_folded)
            else:
                mask = text.str.contains(query_folded, regex=False, na=False)

            matched_rows = mask.to_numpy().nonzero()[0]
            matches.extend((int(row), column) for row in matched_rows)

        return matches


class ExcelSortFilterProxyModel(QSortFilterProxyModel):
    """Proxy model handling sorting and source-row hiding."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._hide_missing_rows = False
        self._missing_rows: set[int] = set()
        self._custom_hidden_rows: set[int] = set()
        self.setDynamicSortFilter(True)

    def set_missing_rows(self, rows: Iterable[int]) -> None:
        self._missing_rows = set(rows)
        self.invalidateFilter()

    def hide_missing_rows(self, hide: bool) -> None:
        self._hide_missing_rows = hide
        self.invalidateFilter()

    def set_hidden_rows(self, rows: Iterable[int]) -> None:
        self._custom_hidden_rows = set(rows)
        self.invalidateFilter()

    def filterAcceptsRow(
            self,
            source_row: int,
            source_parent: QModelIndex,
    ) -> bool:
        if source_row in self._custom_hidden_rows:
            return False
        if self._hide_missing_rows and source_row in self._missing_rows:
            return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        left_value = left.data(Qt.ItemDataRole.UserRole)
        right_value = right.data(Qt.ItemDataRole.UserRole)

        if left_value is None:
            return right_value is not None
        if right_value is None:
            return False

        try:
            return left_value < right_value
        except TypeError:
            return str(left_value).casefold() < str(right_value).casefold()


class FindDialog(QDialog):
    """ Find and replace dialog for :class:`ExcelTable`. """

    def __init__(self, table: 'ExcelTable'):
        super().__init__(table)

        self._table = table
        self._query_matches: list[QModelIndex] = []
        self._match_index = -1
        self._old_query: Optional[str] = None
        self._old_scope: Optional[int] = None
        self._old_exact: Optional[bool] = None

        self._initialize_window()
        self._add_widgets()

    def _initialize_window(self) -> None:
        self.setWindowTitle('Find / Replace')
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.Tool, True)

    def _add_widgets(self) -> None:
        root = QVBoxLayout(self)

        row_one = QHBoxLayout()
        find_label = QLabel('Find:')
        replace_label = QLabel('Replace:')
        label_width = max(find_label.sizeHint().width(), replace_label.sizeHint().width())
        find_label.setFixedWidth(label_width)
        replace_label.setFixedWidth(label_width)
        find_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        replace_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row_one.addWidget(find_label)

        self._line_edit = QLineEdit()
        self._line_edit.setPlaceholderText('Search a keyword...')
        self._line_edit.returnPressed.connect(self._find)
        row_one.addWidget(self._line_edit)

        self._combo_scope = QComboBox()
        self._combo_scope.addItems(['Entire Table'] + self._table.columns)
        self._combo_scope.currentIndexChanged.connect(self._find)
        row_one.addWidget(self._combo_scope)

        self._checkbox_exact_match = QCheckBox('Exact match')
        self._checkbox_exact_match.toggled.connect(self._find)
        row_one.addWidget(self._checkbox_exact_match)

        root.addLayout(row_one)

        row_two = QHBoxLayout()
        row_two.addWidget(replace_label)

        self._replace_edit = QLineEdit()
        self._replace_edit.setPlaceholderText('Replace with...')
        row_two.addWidget(self._replace_edit)

        self._btn_replace = QPushButton('Replace')
        self._btn_replace.clicked.connect(self._replace_current)
        row_two.addWidget(self._btn_replace)

        self._btn_replace_all = QPushButton('Replace All')
        self._btn_replace_all.clicked.connect(self._replace_all)
        row_two.addWidget(self._btn_replace_all)

        root.addLayout(row_two)

        row_three = QHBoxLayout()

        self._label_status = QLabel('0 matches')

        self._btn_prev = QToolButton()
        self._btn_prev.setArrowType(Qt.ArrowType.LeftArrow)
        self._btn_prev.clicked.connect(self._goto_prev)

        self._btn_next = QToolButton()
        self._btn_next.setArrowType(Qt.ArrowType.RightArrow)
        self._btn_next.clicked.connect(self._goto_next)

        self._btn_find = QPushButton('Find')
        self._btn_find.clicked.connect(self._find)

        row_three.addWidget(self._btn_prev)
        row_three.addWidget(self._btn_next)
        row_three.addStretch(1)
        row_three.addWidget(self._label_status)
        row_three.addStretch(1)
        row_three.addWidget(self._btn_find)

        root.addLayout(row_three)

        current_index = self._table.currentIndex()

        if current_index.isValid():
            self._line_edit.setText(str(current_index.data() or ''))

        self._update_status()

    def _find(self) -> None:
        query = self._line_edit.text()
        exact_match = self._checkbox_exact_match.isChecked()
        selected_scope_index = self._combo_scope.currentIndex()

        if not query:
            self._clear_matches()
            return

        if query == self._old_query and exact_match == self._old_exact and selected_scope_index == self._old_scope:
            self._goto_next()
            return

        if selected_scope_index == 0 and not exact_match:
            proceed = QMessageBox.warning(
                self,
                'Potentially Expensive Search',
                'Searching the entire table for a partial match can be '
                'expensive for very large DataFrames. Do you want to proceed?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) == QMessageBox.StandardButton.Yes

            if not proceed:
                return

        self._old_query = query
        self._old_exact = exact_match
        self._old_scope = selected_scope_index

        self._refresh_matches()

    def _refresh_matches(self, preferred_index: int = 0) -> None:
        query = self._line_edit.text()

        if not query:
            self._clear_matches()
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            self._query_matches = self._table.compute_matches(
                query=query,
                exact_match=self._checkbox_exact_match.isChecked(),
                selected_scope_index=self._combo_scope.currentIndex(),
            )
        finally:
            QApplication.restoreOverrideCursor()

        if self._query_matches:
            self._match_index = min(preferred_index, len(self._query_matches) - 1)
        else:
            self._match_index = -1

        self._table.highlight_items(self._query_matches)

        if self._match_index >= 0:
            self._table.focus_item(self._query_matches[self._match_index])

        self._update_status()

    def _clear_matches(self) -> None:
        self._query_matches = []
        self._match_index = -1

        self._table.clear_selection()
        self._update_status()

    def _ensure_current_search(self) -> bool:
        query = self._line_edit.text()
        exact_match = self._checkbox_exact_match.isChecked()
        selected_scope_index = self._combo_scope.currentIndex()

        search_changed = query != self._old_query or exact_match != self._old_exact or selected_scope_index != self._old_scope

        if search_changed:
            self._find()

        return bool(self._query_matches)

    def _replace_current(self) -> None:
        if self._table.readonly or not self._ensure_current_search():
            return

        match = self._query_matches[self._match_index]

        replaced, rejected = self._table.replace_matches(
            items=[match],
            query=self._line_edit.text(),
            replacement=self._replace_edit.text(),
            exact_match=self._checkbox_exact_match.isChecked(),
        )

        if rejected:
            self._show_replacement_warning(rejected)

        if replaced:
            self._refresh_matches(self._match_index)

    def _replace_all(self) -> None:
        if self._table.readonly or not self._ensure_current_search():
            return

        replaced, rejected = self._table.replace_matches(
            items=list(self._query_matches),
            query=self._line_edit.text(),
            replacement=self._replace_edit.text(),
            exact_match=self._checkbox_exact_match.isChecked(),
        )

        self._refresh_matches()

        if rejected:
            self._show_replacement_warning(rejected)

        if replaced:
            self._table.show_status(f'Replaced {replaced} value(s).')

    def _show_replacement_warning(self, rejected: int) -> None:
        QMessageBox.warning(
            parent=self,
            title='Replacement Rejected',
            text=f'{rejected} replacement(s) were rejected because the new '
            'value is incompatible with the target column dtype.',
        )

    def _update_status(self) -> None:
        count = len(self._query_matches)
        can_replace = count > 0 and not self._table.readonly

        self._btn_replace.setEnabled(can_replace)
        self._btn_replace_all.setEnabled(can_replace)

        if count == 0:
            self._label_status.setText('0 matches')
            self._btn_prev.setEnabled(False)
            self._btn_next.setEnabled(False)
            return

        self._label_status.setText(f'{self._match_index + 1} / {count} matches')

        self._btn_prev.setEnabled(True)
        self._btn_next.setEnabled(True)

    def _goto_next(self) -> None:
        if not self._query_matches:
            return

        self._match_index = (self._match_index + 1) % len(self._query_matches)
        self._table.focus_item(self._query_matches[self._match_index])
        self._update_status()

    def _goto_prev(self) -> None:
        if not self._query_matches:
            return

        self._match_index = (self._match_index - 1) % len(self._query_matches)
        self._table.focus_item(self._query_matches[self._match_index])
        self._update_status()


class SuggestedValuesDelegate(QStyledItemDelegate):
    """Use a combobox editor for a column with a finite set of suggestions."""

    def __init__(self, values: Sequence[Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._values = list(values)

    def createEditor(self, parent, _option, _index):
        editor = QComboBox(parent)
        editor.setAutoFillBackground(True)
        for value in self._values:
            editor.addItem(str(value), userData=value)
        editor.activated.connect(lambda _index: self._commit_and_close(editor))
        QTimer.singleShot(0, lambda: self._show_editor(editor))
        return editor

    def setEditorData(self, editor: QComboBox, index: QModelIndex) -> None:
        value = index.data(Qt.ItemDataRole.UserRole)
        selected_index = editor.findData(value)
        if selected_index < 0 and value is not None:
            selected_index = editor.findText(str(value))
        editor.setCurrentIndex(selected_index)

    def setModelData(self, editor: QComboBox, model, index: QModelIndex) -> None:
        if editor.currentIndex() < 0:
            return
        value = editor.currentData()
        model.setData(index, value, Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor: QComboBox, option, _index) -> None:
        cell_rect = option.rect
        editor_rect = QRect(cell_rect)
        editor_rect.setWidth(cell_rect.width() * 2)

        parent = editor.parentWidget()
        if parent is not None and editor_rect.right() > parent.width():
            editor_rect.moveRight(parent.width())

        editor.setGeometry(editor_rect)
        editor.raise_()

    @staticmethod
    def _show_editor(editor: QComboBox) -> None:
        editor.raise_()
        editor.showPopup()

    def _commit_and_close(self, editor: QComboBox) -> None:
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.NoHint)


class ExcelTable(QTableView):
    """ High-performance Excel-like table backed by a pandas DataFrame. The table preserves the behavior of the former
        QTableWidget implementation while avoiding one QTableWidgetItem object per DataFrame cell.

        Attributes
        ----------
        :param parent: (QWidget | None): parent of the table.
        :param df: (pd.DataFrame) dataframe to br displayed.
        :param readonly: (bool) whether table items can be modified.
        :param supports_sorting: (bool) whether sorting can be applied.
        :param supports_query_search: (bool) whether query search is supported.
        :param supports_deletion: (bool) whether deletion of items is supported.
        :param auto_resize: (bool) whether auto-resize of columns is enabled.
                            It can also be enabled/disabled through context menu.
        :param suggested_values: optional values offered through a combobox
                                 editor for each named column.
    """

    def __init__(
            self,
            parent: Optional[QMainWindow],
            df: pd.DataFrame,
            readonly: bool = True,
            supports_sorting: bool = True,
            supports_query_search: bool = True,
            supports_deletion: bool = True,
            auto_resize: bool = False,
            allow_mouse_clicks: bool = True,
            suggested_values: Optional[Mapping[str, Sequence[Any]]] = None,
    ):
        super().__init__(parent)

        self._parent = parent
        self.rows = df.shape[0]
        self.columns = df.columns.tolist()
        self._readonly = readonly
        self._supports_sorting = supports_sorting
        self._supports_query_search = supports_query_search
        self._supports_deletion = supports_deletion
        self._allow_mouse_clicks = allow_mouse_clicks
        self._suggested_values = {
            column: list(values)
            for column, values in (suggested_values or {}).items()
        }

        unknown_suggestion_columns = self._suggested_values.keys() - set(self.columns)
        if unknown_suggestion_columns:
            unknown = ', '.join(sorted(map(str, unknown_suggestion_columns)))
            raise KeyError(f'Unknown suggested-value columns: {unknown}.')

        self._find_dialog = None
        self._toast = None

        self._table_model = DataFrameTableModel(df=df, readonly=readonly, parent=self)
        self._proxy_model = ExcelSortFilterProxyModel(self)
        self._proxy_model.setSourceModel(self._table_model)
        self._proxy_model.set_missing_rows(self._table_model.missing_rows)
        self.setModel(self._proxy_model)

        self._initialize_table()
        self._initialize_suggested_value_delegates()
        self.set_auto_resize(enabled=auto_resize)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._table_model.dataframe

    @property
    def readonly(self) -> bool:
        return self._readonly

    def _initialize_shortcuts(self) -> None:
        """ Initialize persistent keyboard shortcuts for the table window. """

        self._action_copy = QAction('Copy', self)
        self._action_copy.setShortcut(QKeySequence.StandardKey.Copy)
        self._action_copy.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self._action_copy.triggered.connect(self.copy_selection)
        self.addAction(self._action_copy)

        if not self._readonly:
            self._action_paste = QAction('Paste', self)
            self._action_paste.setShortcut(QKeySequence.StandardKey.Paste)
            self._action_paste.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            self._action_paste.triggered.connect(self.paste_selection)
            self.addAction(self._action_paste)

            if self._supports_deletion:
                self._action_delete = QAction('Delete', self)
                self._action_delete.setShortcuts([
                    QKeySequence(Qt.Key.Key_Delete),
                    QKeySequence(Qt.Key.Key_Backspace),
                ])
                self._action_delete.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
                self._action_delete.triggered.connect(self._delete_selection)
                self.addAction(self._action_delete)

            self.selectionModel().selectionChanged.connect(self._update_edit_action_states)
            QApplication.clipboard().dataChanged.connect(self._update_edit_action_states)
            self._update_edit_action_states()

        if self._supports_query_search:
            self._action_find = QAction('Find', self)
            self._action_find.setShortcut(QKeySequence.StandardKey.Find)
            self._action_find.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            self._action_find.triggered.connect(self.open_find_dialog)
            self.addAction(self._action_find)

        self._action_export = QAction('Export Table', self)
        self._action_export.setShortcut(QKeySequence.StandardKey.Save)
        self._action_export.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self._action_export.triggered.connect(self.export_table)
        self.addAction(self._action_export)

        self._action_export_selected = QAction('Export Selected', self)
        self._action_export_selected.triggered.connect(self.export_selected)
        self.addAction(self._action_export_selected)

        self.selectionModel().selectionChanged.connect(self._update_export_action_states)
        self._update_export_action_states()

    def _initialize_table(self):
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
            if self._readonly
            else QAbstractItemView.EditTrigger.DoubleClicked
                 | QAbstractItemView.EditTrigger.EditKeyPressed
                 | QAbstractItemView.EditTrigger.SelectedClicked
        )

        horizontal_header = self.horizontalHeader()
        horizontal_header.setSectionsClickable(True)
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        horizontal_header.setStretchLastSection(False)
        horizontal_header.setResizeContentsPrecision(200)

        vertical_header = self.verticalHeader()
        vertical_header.setSectionsClickable(True)
        vertical_header.sectionClicked.connect(self._select_source_row_from_header)

        if self._supports_sorting:
            horizontal_header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
            self.setSortingEnabled(True)
            horizontal_header.setSortIndicatorShown(True)
        else:
            self.setSortingEnabled(False)
            horizontal_header.setSortIndicatorShown(False)

        # Avoid measuring cell contents while the table is being created. This
        # keeps initial sizing proportional to the number of columns and leaves
        # the header in Interactive mode so users can still drag each section.
        for column, dtype in enumerate(self._table_model.dataframe.dtypes):
            width = 50 if pd.api.types.is_numeric_dtype(dtype) else 120
            self.setColumnWidth(column, width)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

        self._initialize_shortcuts()

    def _initialize_suggested_value_delegates(self) -> None:
        if self._readonly:
            return

        if self._suggested_values:
            self.setEditTriggers(
                QAbstractItemView.EditTrigger.DoubleClicked
                | QAbstractItemView.EditTrigger.EditKeyPressed
            )

        for column, values in self._suggested_values.items():
            if not values:
                continue
            column_id = self.columns.index(column)
            self.setItemDelegateForColumn(
                column_id,
                SuggestedValuesDelegate(values=values, parent=self),
            )

    def _menu(self, pos) -> None:
        menu = QMenu(self)

        menu.addAction(self._action_copy)

        if not self._readonly:
            self._update_edit_action_states()
            menu.addAction(self._action_paste)

            if self._supports_deletion:
                menu.addAction(self._action_delete)

        if self._supports_query_search:
            menu.addAction(self._action_find)

        menu.addAction(self._action_export)
        self._update_export_action_states()
        menu.addAction(self._action_export_selected)

        horizontal_header = self.horizontalHeader()

        action_resize = QAction('Auto-Resizing', self)
        action_resize.setCheckable(True)
        action_resize.setChecked(horizontal_header.sectionResizeMode(0) == QHeaderView.ResizeMode.ResizeToContents)
        action_resize.toggled.connect(self.set_auto_resize)
        action_resize.setToolTip('Enable continuous resize-to-contents. This can reduce performance for very large tables.')

        menu.addSeparator()
        menu.addAction(action_resize)
        menu.exec(self.viewport().mapToGlobal(pos))

    def _update_edit_action_states(self, *_args) -> None:
        """Enable editing actions only when their required input exists."""

        if self._readonly:
            return

        clipboard_text = QApplication.clipboard().text()
        self._action_paste.setEnabled(bool(clipboard_text))

        if self._supports_deletion:
            self._action_delete.setEnabled(self.selectionModel().hasSelection())

    def _update_export_action_states(self, *_args) -> None:
        """Enable selected-row export only while rows are highlighted."""

        self._action_export_selected.setEnabled(self.has_highlighted_rows)

    @property
    def has_highlighted_rows(self) -> bool:
        """Whether at least one table row contains a highlighted cell."""

        return self.selectionModel().hasSelection()

    def highlighted_row_ids(self) -> list[int]:
        """Return highlighted source-row positions in their displayed order."""

        proxy_rows = {
            row
            for selection_range in self.selectionModel().selection()
            for row in range(selection_range.top(), selection_range.bottom() + 1)
        }

        source_rows = []
        for proxy_row in sorted(proxy_rows):
            source_index = self._proxy_model.mapToSource(self._proxy_model.index(proxy_row, 0))
            if source_index.isValid():
                source_rows.append(source_index.row())

        return source_rows

    def export_table(self) -> None:
        """ Prompt for a CSV or XLSX path and export the table data. """

        self.export_dataframe(self._with_displayed_column_labels(self.dataframe))

    def export_selected(self) -> None:
        """Prompt for a path and export highlighted rows only."""

        row_ids = self.highlighted_row_ids()
        if not row_ids:
            return

        selected_df = self.dataframe.iloc[row_ids].copy()
        self.export_dataframe(self._with_displayed_column_labels(selected_df))

    def _with_displayed_column_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` with the table's display-only column labels applied."""

        column_labels = self._table_model.column_labels
        if column_labels == df.columns.tolist():
            return df

        labeled_df = df.copy()
        labeled_df.columns = column_labels
        return labeled_df

    def export_dataframe(self, df: pd.DataFrame) -> None:
        """Prompt for a CSV or XLSX path and export ``df``."""

        filters = 'CSV Files (*.csv);;Excel Files (*.xlsx)'
        downloads_directory = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            'Export Table',
            downloads_directory,
            filters,
            options=QFileDialog.Option.DontConfirmOverwrite
        )

        if not filepath:
            return

        path = Path(filepath).expanduser()
        supported_suffixes = {'.csv', '.xlsx'}

        if path.suffix.lower() in supported_suffixes:
            path = path.with_suffix(path.suffix.lower())
        else:
            suffix = '.xlsx' if selected_filter == 'Excel Files (*.xlsx)' else '.csv'
            path = path.with_suffix(suffix)

        if not path.parent.is_dir():
            return

        filepath = str(path.resolve())
        append = True

        if path.exists():
            choice_dialog = QMessageBox(self)
            choice_dialog.setIcon(QMessageBox.Icon.Question)
            choice_dialog.setWindowTitle('Existing Table')
            choice_dialog.setText('The selected file already exists. Do you want to append the table data or replace the file?')
            append_button = choice_dialog.addButton('Append', QMessageBox.ButtonRole.AcceptRole)
            replace_button = choice_dialog.addButton('Replace', QMessageBox.ButtonRole.DestructiveRole)
            choice_dialog.addButton(QMessageBox.StandardButton.Cancel)
            choice_dialog.setDefaultButton(append_button)
            choice_dialog.exec()

            clicked_button = choice_dialog.clickedButton()
            if clicked_button is append_button:
                append = True
            elif clicked_button is replace_button:
                append = False
            else:
                return

        try:
            error = export_table(df=df, filepath=filepath, append=append)
        except Exception as exception:
            error = str(exception)

        if error is not None:
            QMessageBox.critical(
                self,
                'Save Failed',
                f'Could not export table data, due to error: "{error}".',
            )
            return

        QMessageBox.information(
            self,
            'Table Exported',
            f'Table data are exported in:\n{filepath}',
        )

    def set_auto_resize(self, enabled: bool) -> None:
        mode = QHeaderView.ResizeMode.ResizeToContents if enabled else QHeaderView.ResizeMode.Interactive
        self.horizontalHeader().setSectionResizeMode(mode)

    def replace_dataframe(self, df: pd.DataFrame) -> None:
        """Replace every row without rebuilding the view or its delegates."""

        self._table_model.replace_dataframe(df=df)
        self.rows = df.shape[0]
        self._proxy_model.set_missing_rows(self._table_model.missing_rows)

    def _hide_toast(self) -> None:
        """ Hide the current toast message. """

        if self._toast is None:
            return

        self._toast.deleteLater()
        self._toast = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        delete_selection = self._supports_deletion and not self._readonly and key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)

        if delete_selection:
            self._delete_selection()
            return

        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._allow_mouse_clicks:
            event.ignore()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._allow_mouse_clicks:
            event.ignore()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if not self._allow_mouse_clicks:
            event.ignore()
        else:
            super().mouseDoubleClickEvent(event)

    def copy_selection(self) -> None:
        """ Copy selected ranges as TSV compatible with Excel and text editors. """

        selection = self.selectionModel().selection()

        if selection.isEmpty():
            self.show_status('No text was copied.')
            return

        blocks = []
        for selection_range in selection:
            lines = []
            for row in range(selection_range.top(), selection_range.bottom() + 1):
                values = []
                for column in range(selection_range.left(), selection_range.right() + 1):
                    value = self.model().index(row, column).data()
                    values.append('' if value is None else str(value))
                lines.append('\t'.join(values))
            blocks.append('\n'.join(lines))

        QApplication.clipboard().setText('\n'.join(blocks))
        self.show_status('Copied to clipboard.')

    def paste_selection(self) -> None:
        """ Paste TSV/newline-delimited clipboard data at the current cell. """
        if self._readonly:
            return

        start = self.currentIndex()
        if not start.isValid():
            return

        text = QApplication.clipboard().text()
        if not text:
            return

        rows = [line.split('\t') for line in text.rstrip('\n').splitlines()]
        changed_source_indexes = []

        for row_offset, values in enumerate(rows):
            proxy_row = start.row() + row_offset
            if proxy_row >= self._proxy_model.rowCount():
                break

            for column_offset, value in enumerate(values):
                proxy_column = start.column() + column_offset
                if proxy_column >= self._proxy_model.columnCount():
                    break

                proxy_index = self._proxy_model.index(proxy_row, proxy_column)
                source_index = self._proxy_model.mapToSource(proxy_index)
                if source_index.isValid():
                    self._table_model.setData(source_index, value)
                    changed_source_indexes.append(source_index)

        if changed_source_indexes:
            self._proxy_model.set_missing_rows(self._table_model.missing_rows)
            self.show_status('Pasted from clipboard.')

    def _delete_selection(self) -> None:
        source_indexes = []
        for proxy_index in self.selectionModel().selectedIndexes():
            source_index = self._proxy_model.mapToSource(proxy_index)
            if source_index.isValid():
                source_indexes.append(source_index)

        self._table_model.clear_cells(source_indexes)
        self._proxy_model.set_missing_rows(self._table_model.missing_rows)

    def open_find_dialog(self) -> None:
        """ Initialize or focus the Find dialog. """

        if self._find_dialog is None or not self._find_dialog.isVisible():
            self._find_dialog = FindDialog(self)
            self._find_dialog.show()
            return

        self._find_dialog.activateWindow()
        self._find_dialog.raise_()

    def compute_matches(self, query: Optional[str], exact_match: bool, selected_scope_index: int) -> list[QModelIndex]:
        """Return visible proxy indexes matching a query."""
        self.clearSelection()
        if not query:
            return []

        source_column = None if selected_scope_index <= 0 else selected_scope_index - 1
        source_matches = self._table_model.find_matches(query=query, exact_match=exact_match, source_column=source_column)

        proxy_matches = []
        for source_row, source_column_id in source_matches:
            source_index = self._table_model.index(
                source_row,
                source_column_id,
            )
            proxy_index = self._proxy_model.mapFromSource(source_index)
            if proxy_index.isValid():
                proxy_matches.append(proxy_index)

        proxy_matches.sort(key=lambda index: (index.row(), index.column()))
        return proxy_matches

    def replace_matches(
            self,
            items: Sequence[QModelIndex],
            query: str,
            replacement: str,
            exact_match: bool,
    ) -> Tuple[int, int]:
        """ Replace matched cells and return (replaced, rejected). """

        if self._readonly or not items:
            return 0, 0

        source_indexes = []

        for item in items:
            if not item.isValid():
                continue

            source_index = self._proxy_model.mapToSource(item)

            if source_index.isValid():
                source_indexes.append(source_index)

        replaced = 0
        rejected = 0
        clear_indexes = []

        for source_index in source_indexes:
            current = source_index.data(Qt.ItemDataRole.DisplayRole)
            current = '' if current is None else str(current)

            if exact_match:
                new_value = replacement
            else:
                new_value = re.sub(
                    re.escape(query),
                    lambda _: replacement,
                    current,
                    flags=re.IGNORECASE,
                )

            if new_value == '':
                clear_indexes.append(source_index)
                continue

            if self._table_model.setData(source_index, new_value):
                replaced += 1
            else:
                rejected += 1

        if clear_indexes:
            self._table_model.clear_cells(clear_indexes)
            replaced += len(clear_indexes)

        self._proxy_model.set_missing_rows(
            self._table_model.missing_rows
        )

        return replaced, rejected

    def highlight_items(self, items: Sequence[QModelIndex]) -> None:
        """Highlight proxy-model indexes with fewer selection operations."""
        self.clearSelection()
        if not items:
            return

        by_row: dict[int, list[int]] = {}
        for index in items:
            if index.isValid():
                by_row.setdefault(index.row(), []).append(index.column())

        selection = QItemSelection()
        for row, columns in by_row.items():
            sorted_columns = sorted(set(columns))
            start = previous = sorted_columns[0]
            for column in sorted_columns[1:]:
                if column == previous + 1:
                    previous = column
                    continue
                selection.select(self._proxy_model.index(row, start), self._proxy_model.index(row, previous))
                start = previous = column
            selection.select(self._proxy_model.index(row, start), self._proxy_model.index(row, previous))

        self.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def highlight_rows(self, row_ids: Sequence[int]) -> None:
        """ Highlight rows using stable source-row positions. """

        if self._proxy_model.columnCount() == 0:
            self.clearSelection()
            return

        proxy_rows = set()
        for source_row in row_ids:
            if not 0 <= source_row < self._table_model.rowCount():
                continue

            left_source = self._table_model.index(source_row, 0)
            left_proxy = self._proxy_model.mapFromSource(left_source)

            if not left_proxy.isValid():
                continue

            proxy_rows.add(left_proxy.row())

        if not proxy_rows:
            self.clearSelection()
            return

        selection = QItemSelection()
        last_column = self._proxy_model.columnCount() - 1
        sorted_rows = sorted(proxy_rows)
        range_start = previous_row = sorted_rows[0]

        for proxy_row in sorted_rows[1:]:
            if proxy_row == previous_row + 1:
                previous_row = proxy_row
                continue

            selection.select(
                self._proxy_model.index(range_start, 0),
                self._proxy_model.index(previous_row, last_column),
            )
            range_start = previous_row = proxy_row

        selection.select(
            self._proxy_model.index(range_start, 0),
            self._proxy_model.index(previous_row, last_column),
        )

        self.selectionModel().select(
            selection,
            QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
        )

    def clear_selection(self) -> None:
        self.clearSelection()

    def focus_item(self, item: Optional[QModelIndex]) -> None:
        """ Set current index and center it in the viewport. """

        if item is None or not item.isValid():
            return

        self.setCurrentIndex(item)
        self.scrollTo(item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def hide_missing(self, hide: bool) -> None:
        """ Hides source rows that contain at least one missing value. """

        self._proxy_model.set_missing_rows(self._table_model.missing_rows)
        self._proxy_model.hide_missing_rows(hide)

        if hide:
            self.show_status(message=f'{self._table_model.num_missing} matches are hidden.')
        else:
            self.show_status(message=f'{self._table_model.num_missing} matches are visible.')

    def hide_columns(self, columns: Union[str, Sequence[str]], hide: bool) -> None:
        """ Hide or show one or more columns by column name. """

        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            self.setColumnHidden(self.columns.index(column), hide)

    def set_column_labels(self, labels: Mapping[str, str]) -> None:
        """Set display-only labels without changing logical dataframe columns."""

        self._table_model.set_column_labels(labels)

    def hide_rows(self, row_ids: List[int]) -> None:
        """Replace the hidden rows using source-model row positions.

        Previously hidden rows are restored before the supplied rows are
        hidden. Passing an empty list therefore restores every custom-hidden
        row.
        """
        self._proxy_model.set_hidden_rows(row_ids)

    def modify_columns(self, columns: Sequence[str], data: Sequence[Sequence], rows: Optional[Sequence[int]] = None):
        """ Modify backing DataFrame columns and refresh only changed cells. """

        self._table_model.modify_columns(columns=columns, data=data, rows=rows)
        self._proxy_model.set_missing_rows(self._table_model.missing_rows)

    def modify_rows(self, row_ids: List[int], items: List[List[Any]]) -> None:
        """Modify complete backing DataFrame rows and refresh their cells."""

        self._table_model.modify_rows(row_ids=row_ids, items=items)
        self._proxy_model.set_missing_rows(self._table_model.missing_rows)

    def _select_source_row_from_header(self, proxy_row: int) -> None:
        if proxy_row < 0 or proxy_row >= self._proxy_model.rowCount():
            return

        self.selectRow(proxy_row)

    def show_status(self, message: str, duration: int = 3000) -> None:
        """ Show a temporary toast message over the table. """

        if self._toast is not None:
            self._toast.deleteLater()

        self._toast = QLabel(message, self.viewport())
        self._toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._toast.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._toast.setStyleSheet(
            """
            QLabel {
                color: white;
                background-color: rgb(41, 182, 246);\
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            """
        )

        self._toast.adjustSize()
        x = (self.viewport().width() - self._toast.width()) // 2
        y = self.viewport().height() - self._toast.height() - 20
        self._toast.move(x, y)
        self._toast.show()
        self._toast.raise_()
        QTimer.singleShot(duration, self._hide_toast)
