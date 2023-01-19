from tkinter.ttk import Button
from tkinter.ttk import Treeview
from gui.widgets.tooltip import ToolTip


def validate_float_entry(new_value: str) -> bool:
    if new_value == '':
        return True
    else:
        try:
            float(new_value)
            return True
        except ValueError:
            return False


def validate_float_positive_entry(new_value: str) -> bool:
    if new_value == '':
        return True
    else:
        try:
            if float(new_value) > 0:
                return True
        except ValueError:
            return False



def validate_int_entry(new_value: str) -> bool:
    if new_value == '':
        return True
    else:
        try:
            int(new_value)
            return True
        except ValueError:
            return False


def validate_int_positive_entry(new_value: str) -> bool:
    if new_value == '':
        return True
    else:
        try:
            if int(new_value) > 0:
                return True
        except ValueError:
            return False


def create_tooltip_btn(root, x: int, y: int, text: str):
    button = Button(root, text='?', width=2, takefocus=False)
    button.place(x=x, y=y)

    tooltip = ToolTip(widget=button)
    button.bind('<Enter>', lambda event: tooltip.showtip(text))
    button.bind('<Leave>', lambda event: tooltip.hidetip())
