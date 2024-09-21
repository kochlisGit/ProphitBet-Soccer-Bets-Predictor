import tkinter as tk
from tkinter import ttk


class MultiSelectDropdown(ttk.Frame):
    def __init__(self, parent, x: int, y: int, command=None):
        super().__init__(parent)

        self._command = command

        self.selected_items = {}
        self._display_var = tk.StringVar(value='Select Items')

        self._dropdown_button = ttk.Menubutton(self, textvariable=self._display_var, direction='below')
        self._dropdown_button.grid(row=0, column=0)
        self._menu = tk.Menu(self._dropdown_button, tearoff=False)
        self._dropdown_button.config(menu=self._menu)

        self.place(x=x, y=y)

    def clear(self):
        self.selected_items = {}
        self._menu.delete(0, "end")
        self._display_var.set(value='Select Items')

    def set_items(self, items: list[str]):
        self.clear()

        for item in items:
            item_var = tk.BooleanVar(value=False)
            self.selected_items[item] = item_var
            self._menu.add_checkbutton(
                label=item,
                variable=item_var,
                command=self._on_update
            )

    def get_selected_items(self) -> list:
        return [item for item, var in self.selected_items.items() if var.get()]

    def _on_update(self):
        selected_count = sum(var.get() for var in self.selected_items.values())

        if selected_count:
            self._display_var.set(f'{selected_count} Selected')
        else:
            self._display_var.set('Select Items')

        if self._command is not None:
            self._command()

    def destroy(self):
        self._menu.destroy()
        self._dropdown_button.destroy()
