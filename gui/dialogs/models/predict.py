import pandas as pd
from tkinter import StringVar, messagebox
from tkinter.ttk import Label, Button, Combobox, Separator, Entry
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.dialog import Dialog
from gui.widgets.utils import validate_odd_entry
from models.tasks import ClassificationTask
from preprocessing.dataset import DatasetPreprocessor


class PredictMatchesDialog(Dialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(root=root, title='Predictions', window_size={'width': 500, 'height': 310})

        self._matches_df = matches_df.dropna().reset_index(drop=True)
        self._league_config = league_config
        self._model_repository = model_repository

        self._model_configs = model_repository.get_model_configs(league_id=league_config.league_id)
        self._dataset_preprocessor = DatasetPreprocessor()
        self._tasks = {
            task.name: task for task in [ClassificationTask.Result, ClassificationTask.Over]
            if task.name in self._model_configs and len(self._model_configs[task.name]) > 0
        }
        self._task_predictions = {
            'Result': {0: 'H', 1: 'D', 2: 'A'},
            'Over': {0: 'U(2.5)', 1: 'O(2.5)'}
        }
        self._match_columns = set(self._matches_df.columns)
        self._all_teams = sorted(self._matches_df['Home Team'].unique().tolist())
        self._validate_odd = self.window.register(validate_odd_entry)

        self._task_var = StringVar()
        self._model_id_var = StringVar()
        self._home_team_var = StringVar()
        self._away_team_var = StringVar()
        self._odd_1_var = StringVar(value='1.00')
        self._odd_x_var = StringVar(value='1.00')
        self._odd_2_var = StringVar(value='1.00')

        self._model_cb = None

    def _create_widgets(self):
        Label(self.window, text='Task:', font=('Arial', 11)).place(x=40, y=20)
        task_cb = Combobox(
            self.window, values=list(self._tasks.keys()), width=10, font=('Arial', 10), state='readonly', textvariable=self._task_var
        )
        task_cb.bind('<<ComboboxSelected>>', self._add_models)
        task_cb.place(x=100, y=20)

        Label(self.window, text='Model:', font=('Arial', 11)).place(x=250, y=20)
        self._model_cb = Combobox(
            self.window, width=16, font=('Arial', 10), state='readonly', textvariable=self._model_id_var
        )
        self._model_cb.place(x=320, y=20)

        Separator(self.window, orient='horizontal').place(x=0, y=60, relwidth=1)

        Label(self.window, text='Home Team', font=('Arial', 12)).place(x=110, y=90)
        Combobox(
            self.window, values=self._all_teams, width=18, font=('Arial', 10), state='readonly', textvariable=self._home_team_var
        ).place(x=75, y=125)

        Label(self.window, text='Away Team', font=('Arial', 12)).place(x=305, y=90)
        Combobox(
            self.window, values=self._all_teams, width=18, font=('Arial', 10), state='readonly', textvariable=self._away_team_var
        ).place(x=270, y=125)

        Label(self.window, text='Odd 1', font=('Arial', 12)).place(x=115, y=180)
        Label(self.window, text='Odd X', font=('Arial', 12)).place(x=235, y=180)
        Label(self.window, text='Odd 2', font=('Arial', 12)).place(x=355, y=180)

        Entry(
            self.window,
            width=9,
            font=('Arial', 10, 'bold'),
            validate='key',
            validatecommand=(self._validate_odd, '%P'),
            state='normal' if '1' in self._match_columns else 'disabled',
            textvariable=self._odd_1_var,
        ).place(x=100, y=210)
        Entry(
            self.window,
            width=9,
            font=('Arial', 10, 'bold'),
            validate='key',
            validatecommand=(self._validate_odd, '%P'),
            state='normal' if 'X' in self._match_columns else 'disabled',
            textvariable=self._odd_x_var,
        ).place(x=220, y=210)
        Entry(
            self.window,
            width=9,
            font=('Arial', 10, 'bold'),
            validate='key',
            validatecommand=(self._validate_odd, '%P'),
            state='normal' if '2' in self._match_columns else 'disabled',
            textvariable=self._odd_2_var,
        ).place(x=340, y=210)

        Button(self.window, text='Predict', command=self._predict).place(x=215, y=260)

    def _add_models(self, event):
        self._model_cb['values'] = list(self._model_configs[self._task_var.get()].keys())
        self._model_id_var.set(value='')

    def _predict(self):
        model_id = self._model_id_var.get()
        home_team = self._home_team_var.get()
        away_team = self._away_team_var.get()
        odd_1 = float(self._odd_1_var.get())
        odd_x = float(self._odd_x_var.get())
        odd_2 = float(self._odd_2_var.get())

        for odd, col in zip([odd_1, odd_x, odd_2], ['1', 'X', '2']):
            if col in self._match_columns:
                if odd <= 1.00:
                    messagebox.showerror(parent=self.window, title='Incorrect Configuration', message=f'Odd {col} should be greater than 1.00, got {odd}')
                    return

        if home_team == away_team:
            messagebox.showerror(parent=self.window, title='Incorrect Configuration', message=f'Select Home cannot be the same as Away Team')
            return

        if home_team == '' or away_team == '':
            messagebox.showerror(parent=self.window, title='Incorrect Configuration', message=f'Select Home, Away Teams')
            return

        if model_id == '':
            messagebox.showerror(parent=self.window, title='Incorrect Configuration', message=f'Select model-id to predict this match')
            return

        x = self._dataset_preprocessor.construct_input(
            matches_df= self._matches_df,
            home_team=home_team,
            away_team=away_team,
            odd_1=odd_1,
            odd_x=odd_x,
            odd_2=odd_2
        )
        task = self._task_var.get()
        model_config = self._model_configs[task][model_id]
        model = self._model_repository.load_model(model_config=model_config)
        y_proba = model.predict_proba(x=x)
        y_pred = y_proba.argmax(axis=1)

        prediction = self._task_predictions[task][y_pred[0]]

        result = f'{home_team} vs {away_team}\nPredicted: {prediction} with probabilities '
        for i, prediction_name in enumerate(self._task_predictions[task].values()):
            result += f'{prediction_name}:  {round(y_proba[0][i], 2)}  |  '

        messagebox.showinfo(
            parent=self.window,
            title='Prediction',
            message=result
        )

    def _init_dialog(self):
        messagebox.showinfo(
            parent=self.window,
            title='Upcoming Predictions',
            message='Select task, model, teams and enter odds to predict the upcoming result.'
                    'If odd columns are selected, then enter odds as well.'
        )

    def _get_dialog_result(self):
        return None
