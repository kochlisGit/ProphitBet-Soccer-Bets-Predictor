from tkinter import StringVar
from tkinter.ttk import Label, Combobox, Button
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from models.tasks import ClassificationTask
from gui.dialogs.dialog import Dialog


class DeleteModelDialog(Dialog):
    def __init__(self, root, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(root=root, title='Delete Model', window_size={'width': 300, 'height': 150})

        self._model_repository = model_repository
        self._model_configs = model_repository.get_model_configs(league_id=league_config.league_id)

        self._tasks = {
            task.name: task for task in [ClassificationTask.Result, ClassificationTask.Over]
            if task.name in self._model_configs and len(self._model_configs[task.name]) > 0
        }

        self._task_cb = None
        self._model_ids_cb = None
        self._delete_btn = None

        self._task_var_id = StringVar()
        self._model_id_var = StringVar()

    def _create_widgets(self):
        Label(self.window, text='Task:', font=('Arial', 14)).place(x=20, y=20)
        self._task_cb = Combobox(
            self.window,
            values=list(self._tasks.keys()),
            width=20,
            font=('Arial', 10),
            state='readonly',
            textvariable=self._task_var_id
        )
        self._task_cb.bind('<<ComboboxSelected>>', self._add_model_ids)
        self._task_cb.place(x=100, y=20)

        Label(self.window, text='Model:', font=('Arial', 14)).place(x=20, y=70)
        self._model_ids_cb = Combobox(
            self.window,
            width=20,
            font=('Arial', 10),
            state='readonly',
            textvariable=self._model_id_var
        )
        self._model_ids_cb.place(x=100, y=70)

        delete_btn = Button(
            self.window,
            text='Delete Model',
            state='normal' if len(self._model_configs) > 0 else 'disabled',
            command=self._delete_league
        )
        delete_btn.place(x=120, y=120)

    def _add_model_ids(self, event):
        self._model_ids_cb['values'] = list(self._model_configs[self._task_var_id.get()])

    def _delete_league(self):
        task = self._task_var_id.get()
        model_id = self._model_id_var.get()

        model_config = self._model_configs[task][model_id]
        self._model_repository.delete_model(model_config=model_config)

        self._task_cb.set('')
        self._model_ids_cb.set('')
        self._model_ids_cb['values'] = []

        if task not in self._model_configs:
            del self._tasks[task]
            self._task_cb['values'] = list(self._tasks.keys())

        if len(self._model_configs) == 0:
            self._delete_btn['state'] = 'disabled'

    def _init_dialog(self):
        return

    def _get_dialog_result(self):
        return None
