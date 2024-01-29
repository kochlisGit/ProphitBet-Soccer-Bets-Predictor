from tkinter import Toplevel
from abc import ABC, abstractmethod


class Dialog(ABC):
    def __init__(self, root, title: str, window_size: dict):
        self._window = Toplevel(root)
        self._title = title
        self._window_size = window_size

        self._window.title(title)
        self._window.geometry(f"{window_size['width']}x{window_size['height']}")
        self._window.resizable(False, False)

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
    def _create_widgets(self):
        pass

    @abstractmethod
    def _init_dialog(self):
        pass

    @abstractmethod
    def _get_dialog_result(self):
        pass

    def open(self):
        self._create_widgets()
        self._init_dialog()
        self._window.mainloop()
        return self._get_dialog_result()

    def open_and_wait(self):
        self._create_widgets()
        self._init_dialog()
        self._window.wait_window()
        return self._get_dialog_result()

    def close(self):
        self._window.destroy()
        self._window.quit()
