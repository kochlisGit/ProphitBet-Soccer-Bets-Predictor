from tkinter import Scrollbar, EXTENDED, VERTICAL, END, Listbox
from tkinter.ttk import Scrollbar


class ScrollableListBox:
    def __init__(self, parent, height: int):
        self._listbox = Listbox(parent, height=height, selectmode=EXTENDED)
        self._listbox_scroll = Scrollbar(parent, orient=VERTICAL, command=self._listbox.yview)
        self._listbox['yscrollcommand'] = self._listbox_scroll.set

    @property
    def config(self):
        return self._listbox.config

    def place(self, x: int, y: int):
        self._listbox.place(x=x, y=y)
        self._listbox.update()
        self._listbox_scroll.place(
            x=x + self._listbox.winfo_reqwidth(),
            y=y,
            height=self._listbox.winfo_reqheight()
        )

    def destroy(self):
        self._listbox.destroy()
        self._listbox_scroll.destroy()

    def add_items(self, items: list[str]):
        self._listbox.delete(0, END)
        for i, item in enumerate(items):
            self._listbox.insert(i + 1, item)

    def get_selected_items(self) -> list[str]:
        if self._listbox is not None:
            selected_ids = self._listbox.curselection()
            return [self._listbox.get(ind) for ind in selected_ids]

        return []
