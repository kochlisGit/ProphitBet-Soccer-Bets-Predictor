from tkinter import StringVar, BooleanVar, IntVar, DoubleVar, Widget, Scale
from tkinter.ttk import Combobox, Checkbutton, Label
from gui.widgets.utils import create_tooltip_btn


class TunableWidget:
    def __init__(
            self,
            widget_cls,
            param_values: list or tuple,
            value_variable: StringVar or BooleanVar or IntVar or DoubleVar,
            window,
            name: str,
            description: str or None,
            x: int,
            y: int,
            x_pad: int,
            **widget_params
    ):
        self._param_values = param_values
        self._value_var = value_variable
        self._widget = None

        enabled_state = 'readonly' if widget_cls == Combobox else 'normal'
        self._tune_var = BooleanVar(value=False)
        tune_checkbox = Checkbutton(
            window,
            text='Tune',
            offvalue=False,
            onvalue=True,
            variable=self._tune_var,
            command=lambda: self._widget.config(state='disabled' if self._tune_var.get() else enabled_state)
        )
        tune_checkbox.place(x=x, y=y)
        tune_checkbox.update()
        x += tune_checkbox.winfo_reqwidth() + x_pad

        name_lb = Label(window, text=f'{name}:', font=('Arial', 10))
        name_lb.place(x=x, y=y)
        name_lb.update()
        x += name_lb.winfo_reqwidth() + x_pad

        self._widget = widget_cls(**widget_params)
        self._widget.place(x=x, y=y if not isinstance(self._widget, Scale) else y - 25)
        self._widget.update()
        x += self._widget.winfo_reqwidth() + x_pad

        create_tooltip_btn(window, text=description).place(x=x, y=y)

    @property
    def widget(self) -> Widget:
        return self._widget

    @property
    def param_values(self) -> list or tuple:
        return self._param_values

    def enable(self):
        self._tune_var.set(value=False)
        set_state = 'readaonly' if isinstance(self._widget, Combobox) else 'normal'
        self._widget.config(state=set_state)

    def is_tunable(self) -> bool:
        return self._tune_var.get()

    def uncheck(self):
        self._tune_var.set(value=False)

    def get_value(self):
        return self._value_var.get()

    def set_value(self, value: str):
        self._value_var.set(value=value)
