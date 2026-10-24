"""Singular-module imports for the task dialog classes.

The application historically imports these classes from ``dialogs.tasks``.
Keeping this module as a compatibility facade also supports the singular
``dialogs.task`` path.
"""

from src.gui.components.dialogs.tasks import (
    CancellableProcess,
    SnakeTaskDialog,
    SnakeThreadTaskDialog,
    Task,
    ThreadTask,
)

__all__ = [
    'CancellableProcess',
    'SnakeTaskDialog',
    'SnakeThreadTaskDialog',
    'Task',
    'ThreadTask',
]
