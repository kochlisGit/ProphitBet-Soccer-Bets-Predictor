from tkinter import Toplevel
from abc import ABC, abstractmethod


class Dialog(ABC):
    def __init__(self, master):
        self._window = Toplevel(master)
        self._initialize()

    @abstractmethod
    def _initialize(self):
        pass

    def start(self):
        self._window.mainloop()

    def exit(self):
        self._window.destroy()
        self._window.quit()
