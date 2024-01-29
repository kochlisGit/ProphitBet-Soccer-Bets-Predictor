from tkinter import  StringVar, BooleanVar
from tkinter.ttk import Label, Combobox, Checkbutton, Button
from database.repositories.league import LeagueRepository
from gui.dialogs.dialog import Dialog
from gui.task import TaskDialog


class LoadLeagueDialog(Dialog):
    def __init__(self, root, league_repository: LeagueRepository):
        self._league_repository = league_repository
        self._created_leagues = self._league_repository.get_created_leagues()

        self._matches_df = None
        self._league_config = None

        self._selected_league_id_var = StringVar()
        self._update_league_var = BooleanVar(value=True)

        super().__init__(root=root, title='Load League', window_size={'width': 300, 'height': 200})

    def _create_widgets(self):
        Label(self.window, text='Select League:', font=('Arial', 14)).place(x=90, y=30)
        Combobox(
            self.window,
            values=self._created_leagues,
            width=22,
            font=('Arial', 10),
            state='readonly',
            textvariable=self._selected_league_id_var
        ).place(x=60, y=70)

        Checkbutton(
            self.window,
            text='Check for Updates',
            onvalue=True,
            offvalue=False,
            variable=self._update_league_var
        ).place(x=100, y=110)
        Button(
            self.window,
            text='Load League',
            state='normal' if len(self._created_leagues) > 0 else 'disabled',
            command=self._load_league
        ).place(x=110, y=155)

    def _load_league(self):
        league_id = self._selected_league_id_var.get()

        if self._update_league_var.get():
            load_fn = self._league_repository.update_league
        else:
            load_fn = self._league_repository.load_league

        self._matches_df = TaskDialog(
            master=self.window,
            title=self.title,
            task=load_fn,
            args=(league_id,)
        ).start()
        self._league_config = self._league_repository.get_league_config(league_id=league_id)
        self.close()

    def _init_dialog(self):
        return

    def _get_dialog_result(self, event=None) -> tuple:
        return self._matches_df, self._league_config
