from typing import Optional

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QWidget


class CheckableComboBox(QComboBox):
    checkedItemsChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)

        model = QStandardItemModel(self)
        self.setModel(model)
        model.rowsInserted.connect(self._initialize_check_states)

        self.view().viewport().installEventFilter(self)

    def _initialize_check_states(self, _parent, first: int, last: int) -> None:
        """Make newly inserted items display an unchecked box immediately."""

        model = self.model()
        for row in range(first, last + 1):
            item = model.item(row)
            if item is None or item.data(Qt.ItemDataRole.CheckStateRole) is not None:
                continue

            item.setCheckable(True)
            item.setCheckState(Qt.CheckState.Unchecked)

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802
        if watched is self.view().viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            index = self.view().indexAt(event.position().toPoint())
            if index.isValid():
                item = self.model().itemFromIndex(index)
                item.setCheckState(
                    Qt.CheckState.Unchecked
                    if item.checkState() == Qt.CheckState.Checked
                    else Qt.CheckState.Checked
                )
                self.checkedItemsChanged.emit()
                return True

        return super().eventFilter(watched, event)
