import pandas as pd
import config
from tabulate import tabulate
from tkinter import scrolledtext, END, StringVar, messagebox
from tkinter.ttk import Button, Combobox, Entry, Label, Separator
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.dialog import Dialog
from gui.task import TaskDialog
from gui.widgets.listbox import ScrollableListBox
from gui.widgets.utils import create_tooltip_btn, validate_id_entry
from models.model import ModelConfig
from models.tasks import ClassificationTask
from models.voting import VotingModel
from preprocessing.dataset import DatasetPreprocessor


class VotingModelDialog(Dialog):
    def __init__(
            self,
            root,
            matches_df: pd.DataFrame,
            league_config: LeagueConfig,
            model_repository: ModelRepository
    ):
        super().__init__(root=root, title='Voting Model', window_size={'width': 520, 'height': 540})

        self._matches_df = matches_df.dropna().reset_index(drop=True).iloc[: config.fit_test_size]
        self._league_config = league_config
        self._model_repository = model_repository

        self._dataset_preprocessor = DatasetPreprocessor()
        self._model_configs = self._model_repository.get_model_configs(league_id=league_config.league_id)

        self._tasks = {
            task.name: task for task in [ClassificationTask.Result, ClassificationTask.Over]
            if task.name in self._model_configs and len(self._model_configs[task.name]) > 1
        }

        self._task_var = StringVar()
        self._model_id_var = StringVar(value='voting-model')

        self._forge_btn = None
        self._textarea = None
        self._listbox = None

    def _init_dialog(self):
        if len(self._tasks) == 0:
            messagebox.showerror(
                parent=self.window,
                title='No trained model found',
                message='Voting model combines several models into a single model. Train some models and try again.'
            )
            self._forge_btn['state'] = 'disabled'
        else:
            messagebox.showinfo(
                parent=self.window,
                title='Models Selection',
                message='Select a task and then use CTRL key to combine multiple models from the below listbox'
            )

    def _get_dialog_result(self):
        return None

    def _create_widgets(self):
        Label(
            self.window, text='Task:', font=('Arial', 12)
        ).place(x=45, y=20)
        task_cb = Combobox(
            self.window, values=list(self._tasks.keys()), state='readonly', font=('Arial', 10), textvariable=self._task_var, width=10
        )
        task_cb.bind('<<ComboboxSelected>>', self._add_model_ids)
        task_cb.place(x=105, y=15)

        Label(
            self.window, text='Model ID:', font=('Arial', 12)
        ).place(x=235, y=15)
        Entry(
            self.window,
            width=17,
            font=('Arial', 10),
            textvariable=self._model_id_var
        ).place(x=325, y=20)
        create_tooltip_btn(
            root=self.window,
            text='Unique Model ID that will be stored in database'
        ).place(x=460, y=20)

        Label(
            self.window, text='--- Trained Models ---', font=('Arial', 12)
        ).place(x=200, y=80)
        self._listbox = ScrollableListBox(parent=self.window, height=10)
        self._listbox.place(x=205, y=110)

        Separator(self.window, orient='horizontal').place(x=0, y=50, relwidth=1)

        self._forge_btn = Button(self.window, text='Forge Model', command=self._forge_model)
        self._forge_btn.place(x=230, y=290)
        self._textarea = scrolledtext.ScrolledText(self.window, width=60, height=12, state='disabled')
        self._textarea.place(x=10, y=330)

    def _add_model_ids(self, event):
        self._listbox.add_items(items=self._model_configs[self._task_var.get()].keys())

    def _create_voting_model(self) -> (VotingModel, ModelConfig) or (None, None):
        selected_model_ids = self._listbox.get_selected_items()

        if len(selected_model_ids) < 2:
            messagebox.showerror(
                parent=self.window,
                title='Invalid Model Selection',
                message='At least 2 selected models are required to forge a voting model.'
            )
            return None, None

        task = self._task_var.get()
        model_id = self._model_id_var.get()

        if not validate_id_entry(parent=self.window, text=model_id):
            return None, None

        if task in self._model_configs and model_id in self._model_configs[task]:
            overwrite_result = messagebox.askyesno(
                parent=self.window,
                title='Model Exists',
                message=f'Model "{model_id}" already exists. Do you want to overwrite it?'
            )
            if not overwrite_result:
                return None, None

        model = VotingModel(
            model_id=model_id,
            model_configs=[self._model_configs[task][selected_model_id] for selected_model_id in selected_model_ids],
            model_repository=self._model_repository
        )
        model_config = ModelConfig(
            league_id=self._league_config.league_id,
            model_id=model_id,
            model_cls=VotingModel,
            task=self._tasks[task],
            model_name=model.model_name
        )
        return model, model_config

    def _forge_model(self):
        model, model_config = self._create_voting_model()

        if model is None:
            return

        task = self._tasks[self._task_var.get()]
        model_id = self._model_id_var.get()

        x_test, y_test, _, _ = self._dataset_preprocessor.preprocess_dataset(
            df=self._matches_df,
            task=task,
            fit_normalizer=False,
            normalizer=None,
            sampler=None
        )
        evaluation_dict, classification_report = TaskDialog(
            master=self.window,
            title='Training Model',
            task=model.evaluate,
            args=(x_test, y_test, True)
        ).start()

        results = '--- Evaluation ---' \
                  f'\n{tabulate([list(evaluation_dict.keys()), list(evaluation_dict.values())])} ' \
                  '\n\n--- Classification Report --- ' \
                  f'\n{classification_report}\nModel: {model_id} has been created.'
        proceed_result = messagebox.askyesno(
            parent=self.window,
            title='Fit Results',
            message=results + 'f\n\nDo you want to save the model?'
        )

        if proceed_result:
            self._model_repository.save_model(model=model, model_config=model_config)

        self._show_results(results=results)

    def _show_results(self, results: str):
        self._textarea.config(state='normal')

        self._textarea.delete('1.0', END)
        self._textarea.insert(END, results)

        self._textarea.config(state='disabled')
