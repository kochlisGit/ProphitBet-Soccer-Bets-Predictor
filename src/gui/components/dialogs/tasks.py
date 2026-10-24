import multiprocessing as mp
import pickle
import random
import subprocess
import sys
import threading
import traceback
from multiprocessing.connection import Connection
from typing import Any, Callable
from PySide6.QtCore import QRectF, QTimer, Qt
from PySide6.QtGui import QColor, QCloseEvent, QKeyEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget

from src.gui.widgets.progress import CircularProgressBar


_RESULT_MESSAGE = b'R'
_ERROR_MESSAGE = b'E'


def _process_entry(
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        connection: Connection,
) -> None:
    """Execute a callable in a child process and report its outcome."""

    try:
        result = function(*args, **kwargs)
        # Connection.send() uses multiprocessing's ForkingPickler. PyTorch
        # registers shared-memory reducers with it, so tensors become temporary
        # Windows handles that can disappear when this worker exits. Serializing
        # first with the standard Pickler produces self-contained bytes.
        payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        connection.send_bytes(_RESULT_MESSAGE + payload)
    except BaseException:
        try:
            error = traceback.format_exc().encode('utf-8', errors='replace')
            connection.send_bytes(_ERROR_MESSAGE + error)
        except BaseException:
            pass
    finally:
        connection.close()


class CancellableProcess:
    """ Runs a picklable callable in a cancellable spawned process.

        The callable, its arguments, and its return value must be picklable when
        using the ``spawn`` start method. Module-level functions are preferred.
    """

    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any):
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._context = mp.get_context('spawn')
        self._parent_connection: Connection | None = None
        self._process: mp.Process | None = None
        self._started = False

    @property
    def is_running(self) -> bool:
        """Return whether the worker process is currently alive."""

        return self._process is not None and self._process.pid is not None and self._process.is_alive()

    def start(self) -> None:
        """Start the worker process exactly once."""

        if self._started:
            raise RuntimeError('The process has already been started.')

        self._started = True
        parent_connection, child_connection = self._context.Pipe(duplex=False)
        process = self._context.Process(target=_process_entry, args=(self._function, self._args, self._kwargs, child_connection))
        self._parent_connection = parent_connection
        self._process = process

        try:
            process.start()
        except BaseException:
            child_connection.close()
            parent_connection.close()
            self._parent_connection = None
            self._process = None
            try:
                process.close()
            except (ValueError, AssertionError):
                pass
            raise
        else:
            child_connection.close()

    def poll(self) -> tuple[str, Any] | None:
        """ Return ``('result', value)``, ``('error', traceback)``, or ``None``. """

        if self._process is None or self._parent_connection is None:
            raise RuntimeError('The process has not been started.')

        if self._parent_connection.poll():
            encoded_message = self._parent_connection.recv_bytes()
            self._process.join(timeout=0.2)

            if not encoded_message:
                return 'error', 'The worker process returned an empty response.'

            message_type, payload = encoded_message[:1], encoded_message[1:]
            if message_type == _RESULT_MESSAGE:
                try:
                    return 'result', pickle.loads(payload)
                except BaseException:
                    return 'error', f'Could not deserialize the task result:\n{traceback.format_exc()}'
            if message_type == _ERROR_MESSAGE:
                return 'error', payload.decode('utf-8', errors='replace')
            return 'error', 'The worker process returned an invalid response.'

        if not self._process.is_alive():
            self._process.join()
            return 'error', 'The worker process terminated without returning a result.'

        return None

    def cancel(self) -> None:
        """ Immediately terminate the worker and any processes it spawned. """

        process = self._process

        if process is not None and process.pid is not None and process.is_alive():
            tree_terminated = self._terminate_process_tree(process.pid)

            if not tree_terminated and process.is_alive():
                process.terminate()

            process.join(timeout=1.0)

            if process.is_alive():
                process.kill()
                process.join()

        self.close()

    def close(self) -> None:
        """Close the pipe and release or asynchronously reap the process handle."""

        if self._parent_connection is not None:
            self._parent_connection.close()
            self._parent_connection = None

        process = self._process
        self._process = None

        if process is None:
            return

        if process.pid is None:
            try:
                process.close()
            except (ValueError, AssertionError):
                pass
            return

        if process.is_alive():
            # A result can arrive just before interpreter shutdown completes.
            # Reap that short tail without blocking the Qt event loop.
            threading.Thread(
                target=self._reap_process,
                args=(process,),
                name='TaskProcessReaper',
                daemon=True,
            ).start()
            return

        self._reap_process(process)

    @staticmethod
    def _reap_process(process: mp.Process) -> None:
        """ Wait for a completed worker and release its OS handle. """

        try:
            process.join()
            process.close()
        except (ValueError, AssertionError):
            pass

    @staticmethod
    def _terminate_process_tree(process_id: int) -> bool:
        """ Terminate a complete process tree on Windows when possible. """

        if sys.platform != 'win32':
            return False

        try:
            completed_process = subprocess.run(
                ['taskkill', '/PID', str(process_id), '/T', '/F'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

        return completed_process.returncode == 0


class Task(QDialog):
    """ Display progress while a :class:`CancellableProcess` runs.

        ``run()`` returns the callable result after success and ``None`` after
        cancellation or failure. Worker failures are displayed by the dialog.
    """

    def __init__(
            self,
            function: Callable[..., Any],
            *args: Any,
            title: str = 'Task',
            description: str = 'Please wait while the task is completed.',
            parent: QWidget | None = None,
            **kwargs: Any
    ):
        super().__init__(parent)

        self._runner = CancellableProcess(function, *args, **kwargs)
        self._result: Any = None
        self._error: str | None = None
        self._finished = False
        self._cancelled = False
        self._has_run = False

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._check_process)

        self._setup_dialog(title=title)
        self._setup_ui(description=description)

    @property
    def cancelled(self) -> bool:
        """ Return whether the user cancelled the task. """

        return self._cancelled

    @property
    def error(self) -> str | None:
        """ Return the worker traceback when execution failed. """

        return self._error

    def _setup_dialog(self, title: str) -> None:
        """ Configure the task dialog. """

        self.setWindowTitle(title)
        flags = self.windowFlags()
        flags &= ~Qt.WindowType.WindowMinimizeButtonHint
        flags &= ~Qt.WindowType.WindowMaximizeButtonHint
        flags &= ~Qt.WindowType.WindowContextHelpButtonHint
        flags |= Qt.WindowType.WindowCloseButtonHint
        self.setWindowFlags(flags)
        self.setModal(True)
        self.setMinimumWidth(430)

    def _setup_ui(self, description: str) -> None:
        """ Create the progress-dialog interface. """

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)

        message_label = QLabel(description)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setWordWrap(True)
        layout.addWidget(message_label)

        progress_layout = QHBoxLayout()
        progress_layout.addStretch()
        self._progress_bar = CircularProgressBar(46)
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addStretch()
        layout.addLayout(progress_layout)

    def run(self) -> Any:
        """ Run until completion, failure, or cancellation. """

        if self._has_run:
            raise RuntimeError('Task instance can only be run once.')
        self._has_run = True

        try:
            self._runner.start()
        except BaseException:
            self._error = traceback.format_exc()
            self._finished = True
            self._runner.cancel()
            self._show_critical_error(self._error)
            return None

        self._progress_bar.start()
        self._poll_timer.start()

        try:
            dialog_code = super().exec()
        finally:
            self._stop_activity()
            if not self._finished:
                self._cancelled = True
                self._finished = True
                self._runner.cancel()

        if dialog_code == QDialog.DialogCode.Accepted:
            return self._result
        return None

    def _check_process(self) -> None:
        """ Poll the worker without blocking the Qt event loop. """

        try:
            message = self._runner.poll()
        except (EOFError, OSError):
            message = 'error', 'Communication with the worker process was interrupted.'

        if message is None:
            return

        self._stop_activity()
        status, payload = message
        if status == 'result':
            self._result = payload
            self._finished = True
            self._runner.close()
            super().accept()
            return

        self._result = None
        self._error = str(payload)
        self._finished = True
        self._runner.cancel()
        self._show_critical_error(self._error)
        super().reject()

    def abort(self) -> None:
        """ Cancel the running task immediately. """

        self.reject()

    def reject(self) -> None:
        """ Cancel on Escape or normal dialog rejection. """

        self._cancel_running_task()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        """ Cancel immediately when the dialog's close button is pressed. """

        self._cancel_running_task()
        event.accept()

    def _cancel_running_task(self) -> None:
        """ Stop UI activity and terminate an unfinished worker. """

        if not self._has_run or self._finished:
            return

        self._cancelled = True
        self._finished = True
        self._result = None
        self._stop_activity()
        self._runner.cancel()

    def _stop_activity(self) -> None:
        """ Stop polling and the progress animation. """

        self._poll_timer.stop()
        self._progress_bar.stop()

    def _show_critical_error(self, error: str) -> None:
        """Display a process start, communication, or worker error."""

        QMessageBox.critical(self, 'Process Error', error)


class _SnakeGame(QWidget):
    """Small, continuously playable snake board used as task activity UI."""

    _COLUMNS = 36
    _ROWS = 8
    _CELL_SIZE = 10
    _PADDING = 5
    _INITIAL_LENGTH = 7
    _TICK_INTERVAL_MS = 105

    _DIRECTIONS = {
        Qt.Key.Key_Up: (0, -1),
        Qt.Key.Key_Down: (0, 1),
        Qt.Key.Key_Left: (-1, 0),
        Qt.Key.Key_Right: (1, 0),
    }

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setFixedSize(
            self._COLUMNS * self._CELL_SIZE + self._PADDING * 2,
            self._ROWS * self._CELL_SIZE + self._PADDING * 2,
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._random = random.Random()
        self._timer = QTimer(self)
        self._timer.setInterval(self._TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)

        self._snake: list[tuple[int, int]] = []
        self._food = (0, 0)
        self._direction = (1, 0)
        self._next_direction = self._direction
        self._reset()

    def start(self) -> None:
        """Start moving the snake and direct keyboard input to the board."""

        self._timer.start()
        QTimer.singleShot(0, lambda: self.setFocus(Qt.FocusReason.OtherFocusReason))

    def stop(self) -> None:
        """Stop game activity when its background task is no longer running."""

        self._timer.stop()

    def handle_key(self, key: int) -> bool:
        """Queue an arrow-key direction and return whether it was handled."""

        direction = self._DIRECTIONS.get(key)
        if direction is None:
            return False

        # A direct reversal would make the head collide with the next segment.
        if direction != (-self._direction[0], -self._direction[1]):
            self._next_direction = direction
        return True

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API name
        if self.handle_key(event.key()):
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        board = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        background = self.palette().base().color()
        border = QColor(47, 128, 237, 115 if self.hasFocus() else 70)

        painter.setBrush(background)
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(board, 6.0, 6.0)

        snake_blue = QColor(47, 128, 237)
        for index, (column, row) in enumerate(reversed(self._snake)):
            cell = self._cell_rect(column, row, inset=0.8)
            color = snake_blue.lighter(118) if index == len(self._snake) - 1 else snake_blue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(cell, 2.2, 2.2)

        food = self._cell_rect(*self._food, inset=2.0)
        painter.setPen(QPen(snake_blue, 1.7))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(food)

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().focusOutEvent(event)
        self.update()

    def _cell_rect(self, column: int, row: int, *, inset: float) -> QRectF:
        return QRectF(
            self._PADDING + column * self._CELL_SIZE + inset,
            self._PADDING + row * self._CELL_SIZE + inset,
            self._CELL_SIZE - inset * 2.0,
            self._CELL_SIZE - inset * 2.0,
        )

    def _advance(self) -> None:
        self._direction = self._next_direction
        head_column, head_row = self._snake[0]
        next_head = (
            head_column + self._direction[0],
            head_row + self._direction[1],
        )
        ate_food = next_head == self._food

        hit_wall = not (
            0 <= next_head[0] < self._COLUMNS
            and 0 <= next_head[1] < self._ROWS
        )
        occupied_cells = self._snake if ate_food else self._snake[:-1]
        if hit_wall or next_head in occupied_cells:
            # Keep the activity playable for however long the task takes.
            self._reset()
            self.update()
            return

        self._snake.insert(0, next_head)
        if ate_food:
            if len(self._snake) == self._COLUMNS * self._ROWS:
                self._reset()
            else:
                self._food = self._choose_food_cell()
        else:
            self._snake.pop()
        self.update()

    def _reset(self) -> None:
        self._direction = (1, 0)
        self._next_direction = self._direction
        head_column = self._INITIAL_LENGTH + 1
        head_row = self._ROWS // 2
        self._snake = [
            (head_column - offset, head_row)
            for offset in range(self._INITIAL_LENGTH)
        ]
        self._food = self._choose_food_cell()

    def _choose_food_cell(self) -> tuple[int, int]:
        available_cells = [
            (column, row)
            for row in range(self._ROWS)
            for column in range(self._COLUMNS)
            if (column, row) not in self._snake
        ]
        return self._random.choice(available_cells)


class _SnakeDialogMixin:
    """Replace a task dialog's activity indicator with the snake board."""

    def _setup_ui(self, description: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)

        message_label = QLabel(description)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setWordWrap(True)
        layout.addWidget(message_label)

        self._progress_bar = _SnakeGame(self)
        layout.addWidget(self._progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API name
        if self._progress_bar.handle_key(event.key()):
            event.accept()
            return
        super().keyPressEvent(event)


class SnakeTaskDialog(_SnakeDialogMixin, Task):
    """Run a cancellable process task while the user plays snake."""


class ThreadTask(QDialog):
    """ Run an in-process callable while displaying a modal progress dialog.

        This variant is intended for work such as lazy module imports, where a
        separate process cannot be used because the resulting module state must be
        available to the GUI process. Thread tasks are deliberately not abortable:
        Python cannot safely terminate an active import thread.
    """

    def __init__(
            self,
            function: Callable[..., Any],
            *args: Any,
            title: str = 'Task',
            description: str = 'Please wait while the task is completed.',
            parent: QWidget | None = None,
            **kwargs: Any
    ):
        super().__init__(parent)

        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._result = None
        self._error = None
        self._error_details = None
        self._completed = False
        self._worker_thread = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_thread)

        self.setWindowTitle(title)
        flags = self.windowFlags()
        flags &= ~Qt.WindowType.WindowMinimizeButtonHint
        flags &= ~Qt.WindowType.WindowMaximizeButtonHint
        flags &= ~Qt.WindowType.WindowContextHelpButtonHint
        flags &= ~Qt.WindowType.WindowCloseButtonHint
        self.setWindowFlags(flags)
        self.setModal(True)
        self.setMinimumWidth(430)

        self._setup_ui(description)

    def _setup_ui(self, description: str) -> None:
        """Create the progress-dialog interface."""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)

        message_label = QLabel(description)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setWordWrap(True)
        layout.addWidget(message_label)

        progress_layout = QHBoxLayout()
        progress_layout.addStretch()
        self._progress_bar = CircularProgressBar(46)
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addStretch()
        layout.addLayout(progress_layout)

    def run(self) -> Any:
        """Execute the callable and return its result, or ``None`` on failure."""

        if self._worker_thread is not None or self._completed:
            raise RuntimeError('ThreadTask instance can only be run once.')

        self._worker_thread = threading.Thread(
            target=self._execute,
            name='ThreadTaskWorker',
            daemon=True,
        )
        self._progress_bar.start()
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()
        self._worker_thread.start()
        self._poll_timer.start()

        try:
            super().exec()
        finally:
            self._poll_timer.stop()
            self._progress_bar.stop()
            self._completed = True

        return None if self._error is not None else self._result

    def _execute(self) -> None:
        try:
            self._result = self._function(*self._args, **self._kwargs)
        except BaseException as error:
            self._error = f'{error.__class__.__name__}: {error}'
            self._error_details = traceback.format_exc()

    def _poll_thread(self) -> None:
        if self._worker_thread is None or self._worker_thread.is_alive():
            return

        self._poll_timer.stop()
        self._progress_bar.stop()

        if self._error is None:
            super().accept()
            return

        message = QMessageBox(
            QMessageBox.Icon.Critical,
            'Task failed',
            'The operation could not be completed.',
            parent=self,
        )
        message.setInformativeText(self._error)
        if self._error_details:
            message.setDetailedText(self._error_details)
        message.exec()
        super().reject()

    def reject(self) -> None:
        """Keep the dialog open because an active Python thread cannot be killed safely."""

        if self._worker_thread is None or not self._worker_thread.is_alive():
            super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            event.ignore()
            return
        event.accept()


class SnakeThreadTaskDialog(_SnakeDialogMixin, ThreadTask):
    """Run an in-process thread task while the user plays snake."""
