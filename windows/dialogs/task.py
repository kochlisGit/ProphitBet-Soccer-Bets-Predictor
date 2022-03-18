from tkinter import HORIZONTAL
from tkinter.ttk import Progressbar
from windows.dialogs.dialog import Dialog


class TaskLoaderDialog(Dialog):
    def __init__(self, master, title):
        self._title = title
        self._progressbar = None

        super().__init__(master)

    def _initialize(self):
        self._init_window()
        self._init_progressbar()

    def _init_window(self):
        self._window.title(self._title)
        self._window.geometry('300x200')
        self._window.resizable(False, False)

    def _init_progressbar(self):
        self._progressbar = Progressbar(self._window, orient=HORIZONTAL, length=150, mode='indeterminate')
        self._progressbar.pack(expand=True)

    def start(self):
        self._progressbar.start(8)
        super().start()
