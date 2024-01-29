from tkinter import messagebox
from tkinter.ttk import Button
from gui.widgets.tooltip import ToolTip


def create_tooltip_btn(root, text: str) -> Button:
    button = Button(root, text='?', width=2, takefocus=False)
    tooltip = ToolTip(widget=button)
    button.bind('<Enter>', lambda event: tooltip.showtip(text))
    button.bind('<Leave>', lambda event: tooltip.hidetip())
    return button


def validate_id_entry(parent, text: str) -> bool:
    if not text:
        messagebox.showerror(
            title='Not Accepted ID',
            message='ID is empty'
        )
        return False
    elif not text[0].isalpha():
        messagebox.showerror(
            title='Not Accepted ID',
            message=f'ID should always start with letter (a-z or A-Z), got {text[0]}'
        )
        return False
    else:
        for ch in text:
            if not (ch.isalpha() or ch.isdigit() or ch == '-'):
                messagebox.showerror(
                    title='Not Accepted ID',
                    message=f'Only letters, digits and symbol \'-\' are allowed in ID, got {text}'
                )
                return False
    return True


def validate_odd_entry(new_value: str) -> bool:
    if new_value == '':
        return True
    else:
        try:
            if float(new_value) > 0:
                return True
        except ValueError:
            return False
