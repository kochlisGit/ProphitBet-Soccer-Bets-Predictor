from tkinter import IntVar
from tkinter.ttk import LabeledScale, Label


class PercentileSlider:
    def __init__(self, master, name: str, x: int, y: int, initial_value: int, x_pad: int = 15, command=None):
        self._command = command

        self._variable = IntVar()
        self._label = Label(master, text=name, font=('Arial', 10))
        self._label.place(x=x, y=y)
        self._label.update()

        self._slider = LabeledScale(
            master,
            from_=0,
            to=100,
            variable=self._variable,
            compound='bottom'
        )
        self._slider.scale.bind('<ButtonRelease-1>', self._on_value_change)
        self._slider.place(x=x + x_pad + self._label.winfo_reqwidth(), y=y)
        self._slider.update()
        self._variable.set(value=initial_value)

    @property
    def variable(self) -> IntVar:
        return self._variable

    @property
    def config(self):
        return self._slider.scale.config

    @property
    def slider(self) -> LabeledScale:
        return self._slider

    def update(self):
        self._slider.update()

    def winfo_reqwidth(self):
        return self._label.winfo_reqwidth() + self._slider.winfo_reqwidth()

    def winfo_reqheight(self):
        return self._slider.winfo_reqheight()

    def get_value(self) -> int:
        return self._variable.get()

    def set_value(self, value: int):
        self._variable.set(value=value)

    def destroy(self):
        self._label.destroy()
        self._slider.destroy()
        self._variable = None

    def _on_value_change(self, event):
        value = round(self._variable.get())
        self._variable.set(value)

        if self._command is not None:
            self._command()
