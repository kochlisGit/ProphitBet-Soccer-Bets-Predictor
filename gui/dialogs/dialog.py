from tkinter import Toplevel
from abc import ABC, abstractmethod


class Dialog(ABC):
    def __init__(self, master, title: str, window_size: dict):
        self._window = Toplevel(master)
        self._title = title
        self._window_size = window_size

        self.window.title(title)
        self.window.geometry(f"{window_size['width']}x{window_size['height']}")
        self.window.resizable(False, False)

        self._initialize()

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

    def start(self):
        self._window.mainloop()

    def exit(self):
        self._window.destroy()
        self._window.quit()
