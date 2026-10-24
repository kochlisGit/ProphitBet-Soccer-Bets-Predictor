from typing import Optional
from PySide6.QtCore import QRectF, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class CircularProgressBar(QWidget):
    """ Indeterminate circular progress indicator.

        :param size: (int) progress bar size.
        :param parent (QWidget | nOne): progress bar parent.
    """

    def __init__(self, size: int = 42, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._angle = 0
        self.setFixedSize(size, size)

        self._timer = QTimer(self)
        self._timer.setInterval(35)
        self._timer.timeout.connect(self._rotate)

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _rotate(self):
        self._angle = (self._angle - 12) % 360
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt API name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = 5.5
        border_width = 3.0
        margin = (width + border_width) / 2.0 + 2.0
        rect = QRectF(margin, margin, self.width() - margin * 2.0, self.height() - margin * 2.0)

        window_color = self.palette().window().color()
        foreground = self.palette().highlight().color()
        brightness = window_color.red() * 0.299 + window_color.green() * 0.587 + window_color.blue() * 0.114

        if brightness < 128:
            background = QColor(170, 170, 170)
        else:
            background = QColor(90, 90, 90)

        border = foreground.darker(170)

        pen = QPen(background)
        pen.setWidthF(width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawEllipse(rect)

        pen.setColor(border)
        pen.setWidthF(width + border_width)
        painter.setPen(pen)
        painter.drawArc(rect, self._angle * 16, 110 * 16)

        pen.setColor(foreground)
        pen.setWidthF(width)
        painter.setPen(pen)
        painter.drawArc(rect, self._angle * 16, 110 * 16)
