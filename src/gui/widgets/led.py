from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QLabel, QHBoxLayout, QWidget


class LEDLight(QWidget):
    """ Led widget with 2 states.

        States:
            * Red: indicates not ready state.
            * Green: indicates ready state.

        Attributes
        ----------
        :param ready: (bool) whether state is ready (green) or not (red).
        :param size: (int) size of the widget.
        :param parent: (QWidget | None) parent of the widget.
    """

    def __init__(self, ready: bool = False, size: int = 16, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._ready = ready
        self.setFixedSize(size, size)

    @property
    def ready(self) -> bool:
        return self._ready

    def set_state(self, ready: bool) -> None:
        self._ready = ready
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._ready:
            dark = QColor(0, 120, 20)
            main = QColor(0, 255, 70)
            bright = QColor(190, 255, 210)
            glow = QColor(0, 255, 70, 90)
        else:
            dark = QColor(150, 0, 0)
            main = QColor(255, 30, 30)
            bright = QColor(255, 200, 200)
            glow = QColor(255, 30, 30, 90)

        rect = self.rect().adjusted(2.0, 2.0, -2.0, -2.0)

        # Soft outer glow.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(self.rect().adjusted(0.5, 0.5, -0.5, -0.5))

        # Main LED gradient.
        gradient = QRadialGradient(
            rect.center().x() - rect.width() * 0.20,
            rect.center().y() - rect.height() * 0.20,
            rect.width() * 0.70,
            )
        gradient.setColorAt(0.00, QColor(255, 255, 255))
        gradient.setColorAt(0.15, bright)
        gradient.setColorAt(0.45, main)
        gradient.setColorAt(1.00, dark)

        painter.setPen(QPen(QColor(55, 55, 55), 1.0))
        painter.setBrush(gradient)
        painter.drawEllipse(rect)

        # Strong specular highlight.
        highlight_size = rect.width() * 0.28

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 220))

        painter.drawEllipse(
            rect.left() + rect.width() * 0.18,
            rect.top() + rect.height() * 0.14,
            highlight_size,
            highlight_size * 0.65,
        )


class LedIndicator(QWidget):
    """ LED indicator which includes a LED light and an optional text.

        Attributes
        ----------
        :param ready: (bool) whether state is ready (green) or not (red).
        :param text: (str | None) text of the LED.
        :param size: (int) size of the widget.
        :param parent: (QWidget | None) parent of the widget.
    """

    def __init__(
            self,
            ready: bool = False,
            text: Optional[str] = None,
            size: int = 16,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._led = LEDLight(ready=ready, size=size)
        self._label = QLabel(text or '')
        font = self._label.font()
        font.setPointSize(10)
        self._label.setFont(font)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._led)
        layout.addWidget(self._label)

    @property
    def ready(self) -> bool:
        return self._led.ready

    def set_state(self, ready: bool, text: Optional[str] = None) -> None:
        self._led.set_state(ready=ready)
        self._label.setText(text or '')
