from tkinter import Toplevel
from abc import ABC, abstractmethod


class Dialog(ABC):
    def __init__(self, root, title: str, window_size: dict):
        self._window = Toplevel(root)
        self._title = title
        self._window_size = window_size

        self.window.title(title)
        self.window.geometry(f"{window_size['width']}x{window_size['height']}")
        self.window.resizable(False, False)

    @property
    def window(self):
        return self._window

    @property
    def title(self) -> str:
        return self._title

    @property
    def window_size(self) -> dict:
        return self._window_size

    @abstractmethod
    def _initialize(self):
        pass

    def open(self):
        self._initialize()
        self._window.mainloop()
        return self._dialog_result()

    def close(self):
        self._window.destroy()
        self._window.quit()

    @abstractmethod
    def _dialog_result(self):
        pass
