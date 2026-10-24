from typing import List, Optional
from PySide6.QtCharts import QBarSet
from PySide6.QtCore import QEvent, QEasingCurve, QObject, QVariantAnimation


class BarplotScaling(QObject):
    """ Animate a newly created bar set from zero to its final values once. """

    def __init__(self, bar_set: QBarSet, values: List[float], duration: int = 900, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._bar_set = bar_set
        self._values = [float(value) for value in values]
        self._started = False
        self._start_requested = False
        self._watched_parent = None

        for index in range(min(self._bar_set.count(), len(self._values))):
            self._bar_set.replace(index, 0.0)

        self._animation = QVariantAnimation(self)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setDuration(duration)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._scale_bars)
        self._animation.finished.connect(self._set_final_values)

    def start(self):
        """ Start the entrance animation; subsequent calls are ignored. """

        if self._started or self._start_requested:
            return

        self._start_requested = True
        parent = self.parent()

        if parent is not None and hasattr(parent, 'isVisible') and not parent.isVisible():
            self._watched_parent = parent
            parent.installEventFilter(self)
            return

        self._begin()

    def _begin(self):
        if self._started:
            return

        self._started = True

        if self._watched_parent is not None:
            self._watched_parent.removeEventFilter(self)
            self._watched_parent = None

        self._animation.start()

    def stop(self):
        """ Stop the animation without making it restartable. """

        self._start_requested = False
        self._started = True

        if self._watched_parent is not None:
            self._watched_parent.removeEventFilter(self)
            self._watched_parent = None

        self._animation.stop()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self._watched_parent and event.type() == QEvent.Type.Show:
            self._begin()
        return super().eventFilter(watched, event)

    def _scale_bars(self, progress: float):
        for index, value in enumerate(self._values):
            if index >= self._bar_set.count():
                break
            self._bar_set.replace(index, value * float(progress))

    def _set_final_values(self):
        for index, value in enumerate(self._values):
            if index >= self._bar_set.count():
                break
            self._bar_set.replace(index, value)
