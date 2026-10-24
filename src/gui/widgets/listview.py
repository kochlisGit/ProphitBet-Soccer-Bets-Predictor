from collections.abc import Iterable
from collections.abc import Mapping
from typing import Optional
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget
from src.gui.animations.hover import OnHoverScale


class ListRowButton(QPushButton):
    """ List item displayed as a styled push button.

        Attributes
        ----------
        :param text: (str) text displayed by the button.
        :param icon: (QIcon | None) optional icon displayed to the left of the text.
        :param parent: (QWidget | None) parent of the widget.
    """

    def __init__(
            self,
            text: str,
            icon: Optional[QIcon] = None,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)

        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(QSize(24, 24))

        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(42)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Paint the row independently of the global application stylesheet."""

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.isDown():
            background = QColor('#81c995')
        elif self.underMouse():
            background = QColor('#a8d5b5')
        else:
            background = QColor('#d9e2f3')

        border = QColor('#1f4e79')
        text = QColor('#202020')

        painter.setPen(QPen(border, 3))
        painter.setBrush(background)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)
        painter.setPen(text)
        painter.setFont(self.font())

        content_left = 12
        icon = self.icon()
        if not icon.isNull():
            icon_size = self.iconSize()
            icon_rect = QRect(
                content_left,
                (self.height() - icon_size.height()) // 2,
                icon_size.width(),
                icon_size.height(),
            )
            icon.paint(
                painter,
                icon_rect,
                Qt.AlignmentFlag.AlignCenter,
                QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled,
                QIcon.State.On if self.isChecked() else QIcon.State.Off,
            )
            content_left = icon_rect.right() + 1 + 8

        text_rect = self.rect().adjusted(content_left, 0, -12, 0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())


class ListView(QWidget):
    """ A scrollable list of animated push-button rows.

        Rows use one consistent background color.

        Attributes
        ----------
        :param items: (Iterable[str] | None) items to display.
        :param item_icons: (Mapping[str, str] | None) optional icon path for each item.
        :param parent: (QWidget | None) parent of this widget.
    """

    item_pressed = Signal(str)

    def __init__(
            self,
            items: Optional[Iterable[str]] = None,
            item_icons: Optional[Mapping[str, str]] = None,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._items = []
        self._item_icons = dict(item_icons or {})
        self._buttons = []
        self._hover_wrappers = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll_area.setFrameShape(QScrollArea.Shape.StyledPanel)
        self._scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,)

        self._content_widget = QWidget()

        self._items_layout = QVBoxLayout(self._content_widget)
        self._items_layout.setContentsMargins(6, 6, 6, 6)
        self._items_layout.setSpacing(2)
        self._items_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._scroll_area.setWidget(self._content_widget)

        layout.addWidget(self._scroll_area, 1)
        self.set_items(items or [])

    @property
    def items(self) -> list[str]:
        return list(self._items)

    @property
    def buttons(self) -> list[QPushButton]:
        return list(self._buttons)

    def set_items(self, items: Iterable[str]) -> None:
        """Replace the rows while preserving the supplied item order."""

        self._clear_rows()
        self._items = [str(item) for item in items]

        if not self._items:
            empty_label = QLabel('No entries found: List is empty.')
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setEnabled(False)
            empty_label.setContentsMargins(8, 18, 8, 18)
            self._items_layout.addWidget(empty_label)
            return

        for name in self._items:
            icon_path = self._item_icons.get(name)
            icon = QIcon(icon_path) if icon_path else None
            button = ListRowButton(name, icon=icon)
            button.pressed.connect(lambda current_name=name: self.item_pressed.emit(current_name))

            hover_wrapper = OnHoverScale(button, inset=3)
            hover_wrapper.setFixedHeight(48)
            hover_wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            self._buttons.append(button)
            self._hover_wrappers.append(hover_wrapper)
            self._items_layout.addWidget(hover_wrapper)

    def _clear_rows(self) -> None:
        """Remove all existing rows."""

        while self._items_layout.count():
            layout_item = self._items_layout.takeAt(0)
            widget = layout_item.widget()

            if widget is not None:
                widget.deleteLater()

        self._buttons.clear()
        self._hover_wrappers.clear()
