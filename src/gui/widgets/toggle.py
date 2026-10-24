from typing import Optional
from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QWidget


class ToggleSwitch(QAbstractButton):
    """ Animated blue toggle switch.

        Attributes
        ----------
        :param parent (QWidget | None) parent of the toggle switch.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._position = 3.0

        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(46, 24)

        self._animation = QPropertyAnimation(self, b'position', self)
        self._animation.setDuration(180)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self.toggled.connect(self._animate)

    def sizeHint(self) -> QSize:
        return QSize(46, 24)

    def _animate(self, checked: bool):
        diameter = self.height() - 6
        start = self._position

        if checked:
            end = self.width() - diameter - 3
        else:
            end = 3

        self._animation.stop()
        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def get_position(self) -> float:
        return self._position

    def set_position(self, position: float):
        self._position = position
        self.update()

    position = Property(float, get_position, set_position)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(Qt.PenStyle.NoPen)

        if self.isChecked():
            background = QColor('#2196F3')
        else:
            background = QColor('#9E9E9E')

        painter.setBrush(background)

        rect = QRectF(0, 0, self.width(), self.height())
        radius = self.height() / 2
        painter.drawRoundedRect(rect, radius, radius)

        diameter = self.height() - 6
        knob = QRectF(self._position, 3, diameter, diameter)
        painter.setBrush(QColor('white'))
        painter.drawEllipse(knob)


class AnimatedToggleSwitch(QWidget):
    """ Animated toggle switch with an On/Off label.

        Attributes
        ----------
        :param checked (bool) whether it is set to checked state.
        :param parent (QWidget | None) parent of the toggle switch.
    """

    def __init__(self, checked: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._toggle = ToggleSwitch()

        self._label = QLabel()
        font = self._label.font()
        font.setPointSize(11)
        font.setBold(True)
        self._label.setFont(font)

        layout.addWidget(self._toggle)
        layout.addWidget(self._label)

        self._toggle.toggled.connect(self._update_label)

        self.setChecked(checked)

    def _update_label(self, checked: bool):
        self._label.setText('On' if checked else 'Off')

    def isChecked(self) -> bool:
        return self._toggle.isChecked()

    def setChecked(self, checked: bool):
        self._toggle.setChecked(checked)
        self._update_label(checked)

    @property
    def toggled(self):
        return self._toggle.toggled
