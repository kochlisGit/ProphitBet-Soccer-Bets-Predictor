from math import ceil
from typing import Optional
from PySide6.QtCore import QPointF, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class FloatDoubleRangeSlider(QWidget):
    """A custom two-handle horizontal float slider with increments.

    Attributes
    ----------
    :param minimum: (float) minimum slider value.
    :param maximum: (float) maximum slider value.
    :param step: (float) slider increment.
    :param parent: (QWidget | None) slider's parent.
    """

    valuesChanged = Signal(float, float)
    sliderReleased = Signal()

    def __init__(self, minimum: float,maximum: float, step: float, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        if maximum <= minimum or step <= 0:
            raise ValueError('FloatDoubleRangeSlider requires maximum > minimum and step > 0.')

        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._step = float(step)
        self._steps = ceil((self._maximum - self._minimum) / self._step)

        self._lower_step = 0
        self._upper_step = self._steps

        self._active_handle = 'lower'
        self._hover_handle: Optional[str] = None
        self._dragging = False

        # Setting major tick: one major tick after every five small increments.
        self._major_tick_every = 5

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    @property
    def lower_value(self) -> float:
        return self._from_step(self._lower_step)

    @property
    def upper_value(self) -> float:
        return self._from_step(self._upper_step)

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def maximum(self) -> float:
        return self._maximum

    def _to_step(self, value: float) -> int:
        value = min(self._maximum, max(self._minimum, float(value)))

        if value == self._maximum:
            return self._steps

        return int(round((value - self._minimum) / self._step))

    def _from_step(self, step: int) -> float:
        value = min(self._maximum, self._minimum + step * self._step)
        return round(value, 10)

    def _format_value(self, value: float) -> str:
        decimals = max(0, len(str(self._step).split('.')[-1]))
        return f'{value:.{decimals}f}'

    def _step_x(self, step: int) -> float:
        track = self._track_rect()

        if self._steps == 0:
            return float(track.left())

        ratio = step/self._steps
        return track.left() + ratio * track.width()

    @staticmethod
    def _track_y() -> int:
        return 20

    def _track_rect(self) -> QRect:
        metrics = self.fontMetrics()
        minimum_text = self._format_value(self._minimum)
        maximum_text = self._format_value(self._maximum)
        left_margin = metrics.horizontalAdvance(minimum_text) + 14
        right_margin = metrics.horizontalAdvance(maximum_text) + 14
        return QRect(left_margin, 0, max(0, self.width() - left_margin - right_margin), self.height())

    def _handle_center(self, step: int) -> QPointF:
        return QPointF(self._step_x(step), self._track_y())

    def _handle_rect(self, step: int) -> QRect:
        center = self._handle_center(step)
        radius = 11
        return QRect(round(center.x()) - radius, round(center.y()) - radius, radius * 2, radius * 2)

    def _step_at_position(self, x_position: float) -> int:
        track = self._track_rect()

        if track.width() <= 0:
            return 0

        ratio = (x_position - track.left()) / track.width()
        ratio = min(1.0, max(0.0, ratio))
        return int(round(ratio * self._steps))


    @staticmethod
    def _alpha_color(color: QColor, alpha: int) -> QColor:
        result = QColor(color)
        result.setAlpha(alpha)
        return result

    def _draw_centered_text(self, painter: QPainter, x: float, y: int, text: str) -> None:
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text)

        text_x = round(x - width / 2)
        text_x = max(2, min(text_x, self.width() - width - 2))
        painter.drawText(text_x, y, text)

    def sizeHint(self) -> QSize:
        preferred_width = max(250, min(650, self._steps * 4 + 80))
        return QSize(preferred_width, 50)

    def minimumSizeHint(self) -> QSize:
        return QSize(180, 50)

    def values(self) -> tuple[float, float]:
        return self.lower_value, self.upper_value

    def setRange(self, minimum: float, maximum: float) -> None:
        """Update the slider limits while keeping its values within the new range."""

        minimum = float(minimum)
        maximum = float(maximum)
        if maximum <= minimum:
            raise ValueError('FloatDoubleRangeSlider requires maximum > minimum.')

        lower = self.lower_value
        upper = self.upper_value

        self._minimum = minimum
        self._maximum = maximum
        self._steps = ceil((maximum - minimum) / self._step)
        self.setValues(lower=lower, upper=upper)
        self.updateGeometry()

    def setMinimum(self, minimum: float) -> None:
        self.setRange(minimum=minimum, maximum=self._maximum)

    def setMaximum(self, maximum: float) -> None:
        self.setRange(minimum=self._minimum, maximum=maximum)

    def setValues(self, lower: float, upper: float) -> None:
        lower_step = self._to_step(lower)
        upper_step = self._to_step(upper)

        if lower_step > upper_step:
            lower_step, upper_step = upper_step, lower_step

        self._lower_step = lower_step
        self._upper_step = upper_step
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        palette = self.palette()
        track = self._track_rect()
        y = self._track_y()

        lower_center = self._handle_center(self._lower_step)
        upper_center = self._handle_center(self._upper_step)

        active_color = palette.color(palette.ColorRole.Highlight)
        text_color = palette.color(palette.ColorRole.Text)
        base_color = palette.color(palette.ColorRole.Base)
        mid_color = palette.color(palette.ColorRole.Mid)

        if not self.isEnabled():
            active_color = palette.color(palette.ColorRole.Mid)

        track_color = self._alpha_color(mid_color, 150)
        minor_tick_color = self._alpha_color(text_color, 90)
        major_tick_color = self._alpha_color(text_color, 170)
        active_tick_color = self._alpha_color(active_color, 220)

        # Paining base slider track.
        painter.setPen(QPen(track_color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(track.left(), y, track.right(), y)

        # Paining selected range an intense color.
        painter.setPen(QPen(active_color, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(round(lower_center.x()), y, round(upper_center.x()), y)

        # Paining ticks below.
        minor_tick_top = y + 10
        minor_tick_bottom = y + 14
        major_tick_top = y + 9
        major_tick_bottom = y + 18
        for step_index in range(self._steps + 1):
            x = round(self._step_x(step_index))

            is_major = step_index == 0 or step_index == self._steps or step_index % self._major_tick_every == 0
            is_selected = self._lower_step <= step_index <= self._upper_step

            if is_selected:
                tick_color = active_tick_color
            elif is_major:
                tick_color = major_tick_color
            else:
                tick_color = minor_tick_color

            if is_major:
                pen_width = 2
                top = major_tick_top
                bottom = major_tick_bottom
            else:
                pen_width = 1
                top = minor_tick_top
                bottom = minor_tick_bottom

            painter.setPen(QPen(tick_color, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(x, top, x, bottom)

        # Painting stylized handles.
        handles = (('lower', lower_center), ('upper', upper_center))
        for name, center in handles:
            hovered = name == self._hover_handle
            active = name == self._active_handle

            radius = 10 if hovered or (active and self._dragging) else 9
            shadow_color = self._alpha_color(palette.color(palette.ColorRole.Shadow), alpha=75)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(shadow_color)
            painter.drawEllipse(QPointF(center.x(), center.y() + 2), radius + 1, radius + 1)

            painter.setPen(QPen(active_color, 2))
            painter.setBrush(base_color)
            painter.drawEllipse(center, radius, radius)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(active_color)
            painter.drawEllipse(center, 4, 4)

            if active and self.hasFocus():
                focus_color = self._alpha_color(active_color, 80)

                painter.setPen(QPen(focus_color, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, radius + 4, radius + 4)

        # Displaying current lower and upper values.
        painter.setPen(active_color)

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)

        lower_text = self._format_value(self.lower_value)
        upper_text = self._format_value(self.upper_value)
        selected_label_y = y + 31
        label_gap = 6
        metrics = painter.fontMetrics()
        lower_label_x = lower_center.x() - (metrics.horizontalAdvance(lower_text) + label_gap) / 2
        upper_label_x = upper_center.x() + (metrics.horizontalAdvance(upper_text) + label_gap) / 2
        self._draw_centered_text(painter, lower_label_x, selected_label_y, lower_text)
        self._draw_centered_text(painter, upper_label_x, selected_label_y, upper_text)

        # Displaying minimum and maximum limits.
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(text_color)
        minimum_text = self._format_value(self._minimum)
        maximum_text = self._format_value(self._maximum)

        metrics = painter.fontMetrics()
        maximum_width = metrics.horizontalAdvance(maximum_text)
        text_y = y + metrics.ascent() // 2 - 1
        painter.drawText(2, text_y, minimum_text)
        painter.drawText(self.width() - maximum_width - 2, text_y, maximum_text)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self.isEnabled():
            event.ignore()
            return

        x_position = event.position().x()

        lower_distance = abs(x_position - self._handle_center(self._lower_step).x())
        upper_distance = abs(x_position - self._handle_center(self._upper_step).x())

        if lower_distance == upper_distance:
            midpoint = (self._lower_step + self._upper_step) / 2
            position_step = self._step_at_position(x_position)
            self._active_handle = 'lower' if position_step <= midpoint else 'upper'
        else:
            self._active_handle = 'lower' if lower_distance < upper_distance else 'upper'

        self._dragging = True

        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._move_active_handle(self._step_at_position(x_position))

        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self._move_active_handle(self._step_at_position(event.position().x()))
            event.accept()
            return

        position = event.position().toPoint()
        lower_hover = self._handle_rect(self._lower_step).adjusted(-3, -3, 3, 3).contains(position)
        upper_hover = self._handle_rect(self._upper_step).adjusted(-3, -3, 3, 3).contains(position)
        previous_hover = self._hover_handle

        if lower_hover:
            self._hover_handle = 'lower'
        elif upper_hover:
            self._hover_handle = 'upper'
        else:
            self._hover_handle = None

        if self._hover_handle:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

        if previous_hover != self._hover_handle:
            self.update()

        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        was_dragging = self._dragging
        self._dragging = False
        if was_dragging:
            self.sliderReleased.emit()

        if self._hover_handle:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

        self.update()
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover_handle = None

        if not self._dragging:
            self.unsetCursor()

        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        current_step = self._lower_step if self._active_handle == 'lower' else self._upper_step

        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self._move_active_handle(current_step - 1)
            event.accept()

        elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self._move_active_handle(current_step + 1)
            event.accept()

        elif event.key() == Qt.Key.Key_Home:
            self._move_active_handle(0)
            event.accept()

        elif event.key() == Qt.Key.Key_End:
            self._move_active_handle(self._steps)
            event.accept()

        elif event.key() == Qt.Key.Key_Space:
            self._active_handle = 'upper' if self._active_handle == 'lower' else 'lower'
            self.update()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _move_active_handle(self, new_step: int) -> None:
        new_step = min(self._steps, max(0, new_step))

        if self._active_handle == 'lower':
            new_step = min(new_step, self._upper_step)
            changed = new_step != self._lower_step
            self._lower_step = new_step
        else:
            new_step = max(new_step, self._lower_step)
            changed = new_step != self._upper_step
            self._upper_step = new_step

        if changed:
            self.update()
            self.valuesChanged.emit(self.lower_value, self.upper_value)
