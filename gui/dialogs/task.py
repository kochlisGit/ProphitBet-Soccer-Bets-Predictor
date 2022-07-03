from tkinter import HORIZONTAL
from tkinter.ttk import Progressbar
from gui.dialogs.dialog import Dialog


class TaskLoaderDialog(Dialog):
    def __init__(self, master, title):
        self._progressbar = None

        super().__init__(master, title=title, window_size={'width': 300, 'height': 200})

    def _initialize(self):
        self._init_progressbar()

    def _init_progressbar(self):
        self._progressbar = Progressbar(self.window, orient=HORIZONTAL, length=150, mode='indeterminate')
        self._progressbar.pack(expand=True)

    def start(self):
        self._progressbar.start(8)
        super().start()
