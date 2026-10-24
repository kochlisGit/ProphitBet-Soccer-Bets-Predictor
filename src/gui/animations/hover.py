from typing import Optional
from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, Qt, QVariantAnimation
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QVBoxLayout, QWidget


class OnHoverScale(QWidget):
    """ Provides a subtle scale-up hover animation without modifying the child's geometry directly.
        Instead of scaling the QWidget itself, the surrounding inset is animated:
            * normal: 4 px inset
            * hover:  0 px inset
        The outer widget keeps exactly the same size in the layout, so neighboring widgets do not move.

        Attributes
        ----------
        :param widget: (QWidget) widget that implements the On-Hover Scale animation.
        :param inset: (int) inset value for the widget's animation.
        :param parent: (QWidget | None) parent's widget.
    """

    def __init__(self, widget: QWidget, inset: int = 4, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Enabling hover behavior.
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self._inset = inset
        self._normal_inset = inset

        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.addWidget(widget)

        # Adding hover animation.
        self._animation = QPropertyAnimation(self, b"inset", self)
        self._animation.setDuration(130)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Setting initial margins using the property behavior.
        self.inset = inset

    @Property(int)
    def inset(self) -> int:
        return self._inset

    @inset.setter
    def inset(self, value: int):
        self._inset = value
        self._layout.setContentsMargins(value, value, value, value)

    def _animate_to(self, value: int):
        self._animation.stop()
        self._animation.setStartValue(self._inset)
        self._animation.setEndValue(value)
        self._animation.start()

    def enterEvent(self, event):
        self._animate_to(0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_to(self._normal_inset)
        super().leaveEvent(event)


class OnHoverImageRotation(QWidget):
    """ Provides a constant-rotation hover animation for an image (QPixmap).

        Attributes
        ----------
        :param pixmap: (QPixmap) image to apply the constant rotation.
        :param parent: (QWidget | None) parent's widget.
    """

    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None):
        super().__init__(parent)
        # Setting fixed animation size and enabling hover behavior.
        self.setFixedSize(pixmap.size())
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self._pixmap = pixmap
        self._rotation_angle = 0.0

        self._animation = QVariantAnimation(self)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(360.0)
        self._animation.setDuration(1000)
        self._animation.setLoopCount(-1)
        self._animation.setEasingCurve(QEasingCurve.Type.Linear)
        self._animation.valueChanged.connect(self._on_angle_changed)

    def _on_angle_changed(self, value: float):
        self._rotation_angle = value
        self.update()

    def enterEvent(self, event):
        self._animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animation.stop()
        self._rotation_angle = 0.0
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self.rect()
        painter.translate(rect.center())
        painter.rotate(self._rotation_angle)
        painter.translate(-rect.center())
        painter.drawPixmap(rect, self._pixmap)
