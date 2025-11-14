from typing import List, Optional
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import Qt, QModelIndex


class CheckableComboBox(QComboBox):
    """ Custom Checkable Combobox class. """

    def __init__(self, parent=None, title: Optional[str] = None):
        super().__init__(parent)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)

        if title is not None:
            self.lineEdit().setPlaceholderText(title)

        self.view().pressed.connect(self._handle_item_pressed)

    def addItem(self, text: str):
        """ Adds a single checkable item. """

        super().addItem(text)
        self._make_item_checkable(self.count() - 1)

    def addItems(self, texts: List[str]):
        """ Adds a list of checkable items. """

        for t in texts:
            self.addItem(t)

    def getSelectedIds(self) -> List:
        """ Returns all checked item indices. Excludes 0 (Select All). """

        return [i for i in range(0, self.count()) if self._is_checked(i)]

    def getSelectedTexts(self) -> List:
        """ Returns a list of all selected text of the item. """

        return [self.itemText(i) for i in range(0, self.count()) if self._is_checked(i)]

    # --- internals ------------------------------------------------------------
    def _make_item_checkable(self, i: int):
        """ Constructs a checkable item with an initial unchecked state. """

        item = self.model().item(i, 0)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setCheckState(Qt.CheckState.Unchecked)

    def _is_checked(self, i: int) -> bool:
        """ Returns whether an item is checked. """

        return self.model().item(i, 0).checkState() == Qt.CheckState.Checked

    def _set_checked(self, i: int, checked: bool):
        """ Sets the state of an item as checked/unchecked. """

        self.model().item(i, 0).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _handle_item_pressed(self, index: QModelIndex):
        i = index.row()
        self._set_checked(i, not self._is_checked(i))
