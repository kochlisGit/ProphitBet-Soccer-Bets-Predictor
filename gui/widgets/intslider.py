from tkinter import IntVar
from tkinter.ttk import LabeledScale


class IntSlider:
    def __init__(self, master, from_: int, to: int, variable: IntVar, compound: str = 'bottom'):
        initial_value = variable.get()

        self._slider = LabeledScale(
            master,
            from_=from_,
            to=to,
            variable=variable,
            compound=compound
        )

        if initial_value is not None:
            variable.set(value=initial_value)

        self._slider.scale.bind('<ButtonRelease-1>', self._on_value_change)
        self._variable = variable

    @property
    def config(self):
        return self._slider.scale.config

    @property
    def slider(self) -> LabeledScale:
        return self._slider

    def _on_value_change(self, event):
        value = round(self._variable.get())
        self._variable.set(value)

    def place(self, x: int, y: int):
        self._slider.place(x=x, y=y)

    def destroy(self):
        self._slider.destroy()
        self._slider = None

    def update(self):
        self._slider.update()

    def winfo_reqwidth(self):
        return self._slider.winfo_reqwidth()
