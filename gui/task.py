import threading
from tkinter import Toplevel, HORIZONTAL, messagebox
from tkinter.ttk import Progressbar


class TaskDialog:
    def __init__(self, master, title: str, task, args: tuple or list):
        self._window = Toplevel(master)
        self._title = title
        self._task = task
        self._args = args

        self._event = None
        self._result = None

        def disable_close():
            messagebox.showerror(parent=self._window, title='Cannot Exit', message='You have to wait until the task is finished.')

        self._window.protocol("WM_DELETE_WINDOW", disable_close)

    def start(self):
        self._window.title(self._title)
        self._window.geometry('300x200')
        self._window.resizable(False, False)

        progressbar = Progressbar(self._window, orient=HORIZONTAL, length=150, mode='indeterminate')
        progressbar.pack(expand=True)
        progressbar.start(interval=6)

        threading.Thread(target=self._submit_task, args=()).start()
        self._window.mainloop()
        return self._result

    def _submit_task(self):
        self._result = self._task(*self._args)
        self._window.destroy()
        self._window.quit()
