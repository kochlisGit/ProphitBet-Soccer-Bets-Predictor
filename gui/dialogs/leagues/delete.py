from tkinter import messagebox, StringVar
from tkinter.ttk import Label, Combobox, Button
from database.repositories.league import LeagueRepository
from database.repositories.model import ModelRepository
from gui.dialogs.dialog import Dialog
from gui.task import TaskDialog


class DeleteLeagueDialog(Dialog):
    def __init__(
            self,
            root,
            league_repository: LeagueRepository,
            model_repository: ModelRepository,
            current_league_id: str
    ):
        self._league_repository = league_repository
        self._model_repository = model_repository
        self._created_leagues = self._league_repository.get_created_leagues()
        self._current_league_id = current_league_id

        self._matches_df = None
        self._league_config = None

        self._created_leagues_cb = None
        self._delete_btn = None
        self._selected_league_id_var = StringVar()

        super().__init__(root=root, title='Delete League', window_size={'width': 300, 'height': 150})

    def _create_widgets(self):
        Label(self.window, text='Select League:', font=('Arial', 14)).place(x=90, y=20)
        self._created_leagues_cb = Combobox(
            self.window,
            values=self._created_leagues,
            width=20,
            font=('Arial', 10),
            state='readonly',
            textvariable=self._selected_league_id_var
        )
        self._created_leagues_cb.place(x=70, y=60)
        self._delete_btn = Button(
            self.window,
            text='Delete League',
            state='normal' if len(self._created_leagues) > 0 else 'disabled',
            command=self._delete_league
        )
        self._delete_btn.place(x=110, y=110)

    def _delete_league(self):
        def delete_repositories(league_id: str):
            self._league_repository.delete_league(league_id=league_id)
            self._model_repository.delete_league_models(league_id=league_id)
            self._created_leagues.remove(league_id)

        league_id = self._selected_league_id_var.get()

        if self._current_league_id is not None and self._current_league_id == league_id:
            messagebox.showerror(
                parent=self.window,
                title='League is open',
                message=f'League: {league_id} is currently open. Close the league and try again.'
            )
            return

        TaskDialog(
            master=self.window,
            title=self.title,
            task=delete_repositories,
            args=(league_id,)
        ).start()

        self._created_leagues_cb.set('')
        if len(self._created_leagues) > 0:
            self._created_leagues_cb['values'] = self._created_leagues
        else:
            self._created_leagues_cb['values'] = ['']
            self._delete_btn['state'] = 'disabled'

    def _init_dialog(self):
        return

    def _get_dialog_result(self):
        return None
