import threading
import numpy as np
import pandas as pd
from tkinter import messagebox, StringVar
from tkinter.ttk import Label, Button, Combobox, Entry
from database.repositories.model import ModelRepository
from gui.dialogs.dialog import Dialog
from gui.dialogs.task import TaskDialog
from gui.widgets.utils import validate_float_positive_entry
from models.ensemble import get_ensemble_predictions
from preprocessing.training import construct_input_from_team_names


class PredictionDialog(Dialog):
    def __init__(self, root, matches_df: pd.DataFrame, model_repository: ModelRepository, league_name: str):
        super().__init__(
            root=root,
            title='Prediction',
            window_size={'width': 350, 'height': 300}
        )

        self._matches_df = matches_df
        self._model_repository = model_repository
        self._league_name = league_name
        self._saved_model_names = model_repository.get_all_models(league_name=league_name)

        self._home_team_var = StringVar()
        self._away_team_var = StringVar()

        self._odd_1_var = StringVar()
        self._odd_x_var = StringVar()
        self._odd_2_var = StringVar()
        self._model_name_var = StringVar(value='None')

        self._results_dict = {
            0: 'H', 1: 'D', 2: 'A'
        }

        self._y_pred = None
        self._predict_proba = None

    def _initialize(self):
        all_teams = sorted(self._matches_df['Home Team'].unique().tolist())
        validate_float = self.window.register(validate_float_positive_entry)

        Label(self.window, text='Home Team', font=('Arial', 12)).place(x=45, y=20)
        home_team_cb = Combobox(
            self.window, width=15, font=('Arial', 10), state='readonly', textvariable=self._home_team_var
        )
        home_team_cb['values'] = all_teams
        home_team_cb.current(0)
        home_team_cb.place(x=20, y=55)

        Label(self.window, text='Away Team', font=('Arial', 12)).place(x=225, y=20)
        away_team_cb = Combobox(
            self.window, width=15, font=('Arial', 10), state='readonly', textvariable=self._away_team_var
        )
        away_team_cb['values'] = all_teams
        away_team_cb.current(0)
        away_team_cb.place(x=200, y=55)

        Label(self.window, text='Odd 1', font=('Arial', 12)).place(x=60, y=100)
        Label(self.window, text='Odd X', font=('Arial', 12)).place(x=160, y=100)
        Label(self.window, text='Odd 2', font=('Arial', 12)).place(x=250, y=100)

        Entry(
            self.window, width=9, font=('Arial', 10, 'bold'),
            validate='key', validatecommand=(validate_float, '%P'), textvariable=self._odd_1_var
        ).place(x=45, y=125)

        Entry(
            self.window, width=9, font=('Arial', 10, 'bold'),
            validate='key', validatecommand=(validate_float, '%P'), textvariable=self._odd_x_var
        ).place(x=145, y=125)

        Entry(
            self.window, width=9, font=('Arial', 10, 'bold'),
            validate='key', validatecommand=(validate_float, '%P'), textvariable=self._odd_2_var
        ).place(x=235, y=125)

        Label(self.window, text='Select Model', font=('Arial', 12)).place(x=130, y=170)
        model_name_cb = Combobox(
            self.window, width=12, font=('Arial', 10), state='readonly', textvariable=self._model_name_var
        )

        model_names = self._saved_model_names if len(self._saved_model_names) == 1 \
            else self._saved_model_names + ['Ensemble']

        model_name_cb['values'] = model_names
        model_name_cb.current(0)
        model_name_cb.place(x=120, y=200)

        Button(self.window, text='Predict', command=self._submit_prediction).place(x=135, y=250)

    def _validate_form(self) -> str:
        try:
            odd_1 = float(self._odd_1_var.get())
            if odd_1 <= 1.00:
                return f'Odd_1 is expected to be float number greater than 1.0, got {odd_1}'
        except ValueError:
            return f'Odd_1 is expected to be float number greater than 1.0, got {self._odd_1_var.get()}'

        try:
            odd_x = float(self._odd_x_var.get())
            if odd_x <= 1.00:
                return f'Odd_x is expected to be float number greater than 1.0, got {odd_x}'
        except ValueError:
            return f'Odd_x is expected to be float number greater than 1.0, got {self._odd_x_var.get()}'

        try:
            odd_2 = float(self._odd_2_var.get())
            if odd_2 <= 1.00:
                return f'Odd_2 is expected to be float number greater than 1.0, got {odd_1}'
        except ValueError:
            return f'Odd_2 is expected to be float number greater than 1.0, got {self._odd_2_var.get()}'

        return 'Valid'

    def _predict(self, task_dialog: TaskDialog):
        x = construct_input_from_team_names(
            matches_df=self._matches_df,
            home_team=self._home_team_var.get(),
            away_team=self._away_team_var.get(),
            odd_1=float(self._odd_1_var.get()),
            odd_x=float(self._odd_x_var.get()),
            odd_2=float(self._odd_2_var.get())
        )

        model_name = self._model_name_var.get()

        if model_name == 'Ensemble':
            models = [
                self._model_repository.load_model(
                    league_name=self._league_name, model_name=name, input_shape=x.shape[1:], random_seed=0
                )
                for name in self._saved_model_names
            ]
            y_pred, predict_proba = get_ensemble_predictions(x=x, models=models)
        else:
            model = self._model_repository.load_model(
                league_name=self._league_name, model_name=model_name, input_shape=x.shape[1:], random_seed=0
            )
            y_pred, predict_proba = model.predict(x=x)

        self._y_pred = y_pred[0]
        predict_proba = predict_proba.flatten()
        self._predict_proba = np.round(predict_proba, 2)
        task_dialog.close()

    def _submit_prediction(self):
        validation_result = self._validate_form()

        if validation_result == 'Valid':
            task_dialog = TaskDialog(self._window, self._title)
            task_thread = threading.Thread(target=self._predict, args=(task_dialog,))
            task_thread.start()
            task_dialog.open()

            messagebox.showinfo(
                f'{self._model_name_var.get()} Prediction',
                f'Predicted: {self._results_dict[self._y_pred]},\n'
                f'Probabilities: '
                f'H: {self._predict_proba[0]}%, D: {self._predict_proba[1]}%, A: {self._predict_proba[2]}%'
            )
        else:
            messagebox.showerror('Form Validation Error', validation_result)

    def _dialog_result(self) -> None:
        return None
