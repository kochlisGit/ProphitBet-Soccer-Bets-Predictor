from typing import Any, Callable, Optional, Sequence
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout, QWidget


class ThreadWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn: Callable, args: Sequence[Any] = (), kwargs: Optional[dict] = None):
        super().__init__()

        self._fn = fn
        self._args = tuple(args) if args else ()
        self._kwargs = dict(kwargs) if kwargs else {}

    @pyqtSlot()
    def run(self):
        try:
            res = self._fn(*self._args, **self._kwargs)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class TaskRunnerDialog(QDialog):
    """ Task Runner dialog that blocks the parent application until a task is finished. """

    def __init__(
            self,
            title: str,
            info: str,
            task_fn: Callable,
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._width = 300
        self._height = 80

        self._initialize_window(title=title)
        self._add_widgets(info=info)

        # Thread + worker
        self._thread = QThread(self)
        self._worker = ThreadWorker(task_fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # Cleanup
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

    def _initialize_window(self, title: str):
        self.setWindowTitle(title)
        self.setModal(True)     # Set focus and block parent.
        self.setFixedSize(self._width, self._height)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

    def _add_widgets(self, info: str):
        layout = QVBoxLayout(self)
        self._label = QLabel(info)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # Set progressbar in indeterminate state.
        self._progress.setTextVisible(False)
        layout.addWidget(self._label)
        layout.addWidget(self._progress)

    def run(self) -> Any:
        """Start the task and enter a modal loop. Returns the task result (also stored in self.result)."""
        self._thread.start()
        self.exec()               # modal; keeps UI responsive while worker runs
        return self.result

    def _teardown(self):
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    def _on_finished(self, result: object):
        self.result = result
        self._teardown()
        self.accept()  # close dialog (success)

    def _on_error(self, message: str):
        self.error = message
        self._teardown()
        self.reject()  # close dialog (error)
