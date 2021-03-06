from database.preprocessing import preprocessing
from models.model import Model
from models.neuralnet.nn import FCNet
from models.randomforest.rf import RandomForest
from filters.evaluation import EvaluationFilter
from gui.dialogs.dialog import Dialog
from fixtures.parsers.fixtures import FixtureParser
from fixtures.similarities.matching import TeamSimilarityMatching
from abc import ABC, abstractmethod
from tkinter import messagebox, Text, StringVar, DISABLED, NORMAL, END, CENTER, VERTICAL, HORIZONTAL
from tkinter.ttk import Label, Entry, Button, Treeview, Combobox, Scrollbar
from tkinter import filedialog
import numpy as np
import pandas as pd
import csv
import threading
import ast


class TrainDialog(Dialog, ABC):
    def __init__(
            self,
            master,
            title: str,
            window_size: dict,
            checkpoint_path: str,
            league_identifier: str,
            results_and_stats: pd.DataFrame
    ):
        self._checkpoint_path = checkpoint_path
        self._model_name = league_identifier
        self._results_and_stats = results_and_stats

        self._model = None
        self._training_on_progress = False

        self._text_area = None

        super().__init__(master=master, title=title, window_size=window_size)

    @property
    def results_and_stats(self) -> pd.DataFrame:
        return self._results_and_stats

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    @property
    def league_identifier(self) -> str:
        return self._model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model(self) -> Model:
        return self._model

    def _initialize(self):
        self._initialize_form()

    @abstractmethod
    def _initialize_form(self):
        pass

    @abstractmethod
    def _validate_form(self) -> str or None:
        pass

    def _report_training_log(self, text):
        self._text_area.config(state=NORMAL)
        self._text_area.insert(END, text)
        self._text_area.config(state=DISABLED)

    def _clear_training_log(self):
        self._text_area.config(state=NORMAL)
        self._text_area.delete('1.0', END)
        self._text_area.config(state=DISABLED)

    def _train(self):
        form_validation_result = self._validate_form()

        if form_validation_result == 'Valid':
            if self._training_on_progress:
                messagebox.showerror('Training ERROR', 'Another training is on progress.')
            else:
                train_thread = threading.Thread(target=self._train_model)
                train_thread.start()
        else:
            messagebox.showerror('Validation ERROR', form_validation_result)

    @abstractmethod
    def _train_model(self):
        pass


class TrainNNDialog(TrainDialog):
    def __init__(self, master, checkpoint_path: str, league_identifier: str, results_and_stats: pd.DataFrame):
        self._hidden_layers = [128, 128]
        self._validation_size = 50
        self._odd_noise_range = 0.15
        self._epochs = 200
        self._patience = 50

        self._layers_var = StringVar()
        self._validation_var = StringVar()
        self._noise_var = StringVar()
        self._performance_rate_noise = StringVar()
        self._epochs_var = StringVar()
        self._patience_var = StringVar()

        super().__init__(
            master=master,
            title='Training Neural Network',
            window_size={'width': 400, 'height': 640},
            checkpoint_path=checkpoint_path,
            league_identifier=league_identifier,
            results_and_stats=results_and_stats
        )

    def _initialize_form(self):
        Label(self._window, text='Model\'s Name', font=('Arial', 14)).place(x=30, y=20)
        Label(self._window, text='Hidden Layers', font=('Arial', 14)).place(x=30, y=60)
        Label(self._window, text='Validation Size', font=('Arial', 14)).place(x=30, y=100)
        Label(self._window, text='Noise Range', font=('Arial', 14)).place(x=30, y=140)
        Label(self._window, text='Win/Draw % Noise', font=('Arial', 14)).place(x=30, y=180)
        Label(self._window, text='Epochs', font=('Arial', 14)).place(x=30, y=220)
        Label(self._window, text='Patience', font=('Arial', 14)).place(x=30, y=260)

        name_entry = Entry(self._window, width=15, font=('Arial', 10))
        name_entry.insert(0, self.model_name)
        name_entry.config(state=DISABLED)
        name_entry.place(x=220, y=20)

        layers_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._layers_var)
        layers_entry.insert(0, str(self._hidden_layers))
        layers_entry.place(x=220, y=60)

        validation_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._validation_var)
        validation_entry.insert(0, str(self._validation_size))
        validation_entry.place(x=220, y=100)

        noise_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._noise_var)
        noise_entry.insert(0, str(self._odd_noise_range))
        noise_entry.place(x=220, y=140)

        performance_noise_cb = Combobox(
            self._window, width=13, font=('Arial', 10), textvariable=self._performance_rate_noise
        )
        performance_noise_cb['values'] = ['True', 'False']
        performance_noise_cb.current(0)
        performance_noise_cb.place(x=220, y=180)

        epochs_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._epochs_var)
        epochs_entry.insert(0, str(self._epochs))
        epochs_entry.place(x=220, y=220)

        patience_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._patience_var)
        patience_entry.insert(0, str(self._patience))
        patience_entry.place(x=220, y=260)

        Button(self._window, text='Train', command=self._train).place(x=150, y=300)

        self._text_area = Text(self._window, state=DISABLED, font=('Arial', 10))
        self._text_area.place(x=0, y=340)

    def _validate_form(self) -> str or None:
        layers_str = self._layers_var.get()
        validation_size = self._validation_var.get()
        noise = self._noise_var.get()
        noise = noise.replace('.', '')
        epochs = self._epochs_var.get()
        patience = self._patience_var.get()

        try:
            ast.literal_eval(layers_str) == list
        except:
            return 'List should be in form "[l1, l2, l3, ...]"'
        if not validation_size.isdigit() or int(validation_size) < 0:
            return 'Validation Size should be a positive number'
        if not noise.isdigit() or float(noise) < 0:
            return 'Noise Range should have a positive value < 1.0'
        if not epochs.isdigit() or int(epochs) < 0:
            return 'Epochs should be a positive number'
        if not patience.isdigit() or int(patience) < 0:
            return 'Patience should be a positive number'
        return 'Valid'

    def _train_model(self):
        self._training_on_progress = True
        default_train_log = 'Current Epoch: {}, Validation Accuracy = {}\n'

        hidden_layers = ast.literal_eval(self._layers_var.get())
        validation_size = int(self._validation_var.get())
        noise_range = float(self._noise_var.get())
        performance_rate_noise = bool(self._performance_rate_noise.get())
        epochs = int(self._epochs_var.get())
        patience = int(self._patience_var.get())

        self._report_training_log('Preprocessing data...\n')

        inputs, targets = preprocessing.preprocess_data(self._results_and_stats, one_hot=True)
        x_test = inputs[:validation_size]
        y_test = targets[:validation_size]
        x_train = inputs[validation_size:]
        y_train = targets[validation_size:]

        self._report_training_log('Building the model...\n')

        model = FCNet(
            input_shape=x_train.shape[1:],
            checkpoint_path=self.checkpoint_path,
            league_identifier=self.league_identifier,
            model_name=self.model_name
        )
        model.build_model(hidden_layers=hidden_layers)

        current_epoch = 0
        best_accuracy = 0
        patience_counter = 0

        self._report_training_log('Ready to train...\n')

        while current_epoch < epochs and patience_counter < patience:
            accuracy = model.train(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                odd_noise_range=noise_range,
                performance_rate_noise=performance_rate_noise
            )

            self._report_training_log(default_train_log.format(
                current_epoch,
                accuracy,
                best_accuracy
            ))

            if accuracy > best_accuracy:
                self._report_training_log('New best model is found! Saving the model...\n')

                best_accuracy = accuracy
                patience_counter = 0
                model.save()
            else:
                patience_counter += 1
            current_epoch += 1

            if current_epoch % 10 == 0:
                self._clear_training_log()
                self._report_training_log('Best accuracy: so far: {}. Continuing...'.format(best_accuracy))

        self._clear_training_log()
        self._report_training_log('Training has finished with best accuracy: {} after {} epochs'.format(
            best_accuracy,
            current_epoch
        ))
        self._training_on_progress = False


class TrainRFDialog(TrainDialog):
    def __init__(self, master, checkpoint_path: str, league_identifier: str, results_and_stats: pd.DataFrame):
        self._n_estimators = 100
        self._calibration = True
        self._validation_size = 50

        self._n_estimators_var = StringVar()
        self._calibration_var = StringVar()
        self._validation_var = StringVar()

        super().__init__(
            master=master,
            title='Training Random Forest',
            window_size={'width': 400, 'height': 600},
            checkpoint_path=checkpoint_path,
            league_identifier=league_identifier,
            results_and_stats=results_and_stats
        )

    def _initialize_form(self):
        Label(self._window, text='Model\'s Name', font=('Arial', 16)).place(x=30, y=20)
        Label(self._window, text='Estimators', font=('Arial', 16)).place(x=30, y=60)
        Label(self._window, text='Calibration', font=('Arial', 16)).place(x=30, y=100)
        Label(self._window, text='Validation Size', font=('Arial', 16)).place(x=30, y=140)

        name_entry = Entry(self._window, width=15, font=('Arial', 10))
        name_entry.insert(0, self.model_name)
        name_entry.config(state=DISABLED)
        name_entry.place(x=200, y=20)

        estimators_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._n_estimators_var)
        estimators_entry.insert(0, str(self._n_estimators))
        estimators_entry.place(x=200, y=60)

        calibration_cb = Combobox(self._window, width=15, font=('Arial', 10), textvariable=self._calibration_var)
        calibration_cb['values'] = ['True', 'False']
        calibration_cb.current(0)
        calibration_cb.place(x=200, y=100)

        validation_size_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._validation_var)
        validation_size_entry.insert(0, str(self._validation_size))
        validation_size_entry.place(x=200, y=140)

        Button(self._window, text='Train', command=self._train).place(x=150, y=260)

        self._text_area = Text(self._window, state=DISABLED, font=('Arial', 10))
        self._text_area.place(x=0, y=300)

    def _validate_form(self) -> str or None:
        n_estimators = self._n_estimators_var.get()
        validation_size = self._validation_var.get()

        if not n_estimators.isdigit() or int(n_estimators) < 0:
            return 'Number of Estimators should be a positive number'
        elif not validation_size.isdigit() or int(validation_size) < 0:
            return 'Validation Size should be a positive number'
        return 'Valid'

    def _train_model(self):
        self._training_on_progress = True

        n_estimators = int(self._n_estimators_var.get())
        calibration = bool(self._calibration_var.get())
        validation_size = int(self._validation_var.get())

        self._clear_training_log()
        self._report_training_log('Preprocessing data...\n')

        inputs, targets = preprocessing.preprocess_data(self._results_and_stats, one_hot=False)
        x_test = inputs[:validation_size]
        y_test = targets[:validation_size]
        x_train = inputs[validation_size:]
        y_train = targets[validation_size:]

        self._report_training_log('Building the model...\n')

        model = RandomForest(
            input_shape=(),
            checkpoint_path=self.checkpoint_path,
            league_identifier=self.league_identifier,
            model_name=self.model_name,
            calibrate_model=calibration
        )
        model.build_model(n_estimators=n_estimators)

        self._report_training_log('Ready to train...\n')

        accuracy = model.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )
        model.save()

        self._report_training_log(f'Training has finished with accuracy: {accuracy}')
        self._training_on_progress = False


class EvaluationDialog(Dialog):
    def __init__(
            self,
            master,
            models: dict,
            results_and_stats: pd.DataFrame,
            repository_basic_columns: list
    ):
        self._results_and_stats = results_and_stats
        self._models = models

        self._evaluation_columns = [
            'Prediction',
            '1 Prob %',
            'X Prob %',
            '2 Prob %'
        ]

        self._inputs = None
        self._targets = None
        self._evaluation_filter = EvaluationFilter()

        self._last_n_values = [10, 25, 50, 100, 150, 200]
        self._result_filter_values = ['All', '1', 'X', '2']
        self._repository_basic_columns = repository_basic_columns

        self._n_validation_matches_var = StringVar()
        self._result_filter_var = StringVar()
        self._selected_model_var = StringVar()

        self._treeview = None
        self._accuracy_label = None

        super().__init__(master=master, title='Evaluation', window_size={'width': 1050, 'height': 780})

    @property
    def repository_basic_columns(self) -> list:
        return self._repository_basic_columns

    @property
    def treeview_columns(self) -> list:
        return self.repository_basic_columns + self._evaluation_columns

    def _initialize(self):
        self._initialize_form()

    def _initialize_form(self):
        Label(self._window, text='Last N:', font=('Arial', 12)).place(x=50, y=10)

        leagues_cb = Combobox(
            self._window, state='readonly', width=10, font=('Arial', 10), textvariable=self._n_validation_matches_var
        )
        leagues_cb['values'] = self._last_n_values
        leagues_cb.place(x=120, y=10)
        leagues_cb.bind('<<ComboboxSelected>>', self._evaluate_matches)

        Label(self._window, text='Odds Filter:', font=('Arial', 12)).place(x=250, y=10)

        result_filter_cb = Combobox(
            self._window, state='readonly', width=10, font=('Arial', 10), textvariable=self._result_filter_var
        )
        result_filter_cb['values'] = self._result_filter_values
        result_filter_cb.current(0)
        result_filter_cb.place(x=350, y=10)
        result_filter_cb.bind('<<ComboboxSelected>>', self._evaluate_matches)

        Button(self._window, text='Show Accuracy Details', command=self._show_accuracy_details).place(x=500, y=10)

        Label(self._window, text='Accuracy:', font=('Arial', 12)).place(x=700, y=10)
        self._accuracy_label = Label(self._window, font=('Arial', 10))
        self._accuracy_label.place(x=770, y=13)

        self._treeview = Treeview(
            self._window,
            columns=self.treeview_columns,
            show='headings',
            selectmode='browse',
            height=30
        )
        for column_name in self.treeview_columns:
            self._treeview.column(column_name, anchor=CENTER, stretch=True, width=70)
            self._treeview.heading(column_name, text=column_name, anchor=CENTER)
        self._treeview.column('Date', anchor=CENTER, stretch=True, width=100)
        self._treeview.column('Home Team', anchor=CENTER, stretch=True, width=100)
        self._treeview.column('Away Team', anchor=CENTER, stretch=True, width=100)
        self._treeview.place(x=25, y=50)

        v_scroll = Scrollbar(self._window, orient=VERTICAL, command=self._treeview.yview)
        v_scroll.place(x=1030, y=50, height=620)
        self._treeview.configure(yscroll=v_scroll.set)

        h_scroll = Scrollbar(self._window, orient=HORIZONTAL, command=self._treeview.xview)
        h_scroll.place(x=25, y=680, width=1000)
        self._treeview.configure(xscroll=h_scroll.set)

        Label(self._window, text='Load Trained Model', font=('Arial', 12)).place(x=380, y=700)

        models_cb = Combobox(
            self._window, state='readonly', width=20, font=('Arial', 10), textvariable=self._selected_model_var
        )
        models_cb['values'] = list(self._models.keys())
        models_cb.current(0)
        models_cb.place(x=550, y=700)
        models_cb.bind('<<ComboboxSelected>>', self._evaluate_matches)

    def _evaluate_matches(self, event):
        _n_validation_matches_str = self._n_validation_matches_var.get()

        if _n_validation_matches_str == '':
            return

        n_validation_matches = int(_n_validation_matches_str)
        model_checkpoint = self._selected_model_var.get()

        model = self._models[model_checkpoint]
        use_one_hot = False if 'rf' in model_checkpoint else True
        evaluation_results_and_stats = self._results_and_stats.iloc[:n_validation_matches]
        self._inputs, self._targets = preprocessing.preprocess_data(
            results_and_stats=evaluation_results_and_stats, one_hot=use_one_hot
        )
        predict_proba, self._predictions = model.predict(x_inputs=self._inputs)

        if use_one_hot:
            self._targets = np.argmax(self._targets, axis=1)

        self._display_treeview_items(
            evaluation_matches=evaluation_results_and_stats,
            predict_proba=predict_proba,
            predictions=self._predictions
        )
        self._display_validation_accuracy(
            y_targets=self._targets,
            predictions=self._predictions
        )

    def _display_treeview_items(
            self,
            evaluation_matches: pd.DataFrame,
            predict_proba: np.ndarray,
            predictions: np.ndarray
    ):
        for item in self._treeview.get_children():
            self._treeview.delete(item)

        items = []
        predicted_results = preprocessing.predictions_to_result(predictions)

        basic_data = evaluation_matches[self.repository_basic_columns].to_numpy().tolist()
        for i, pred in enumerate(predict_proba):
            items.append(
                basic_data[i] +
                [predicted_results[i]] +
                [round(pred[j], 2) for j in range(3)]
            )

        for i, values in enumerate(items):
            self._treeview.insert(parent='', index=i, values=values)

    def _display_validation_accuracy(self, y_targets: np.ndarray, predictions: np.ndarray):
        correct_predictions = sum(y_targets[i] == predictions[i] for i in range(predictions.shape[0]))
        avg_accuracy = 100 * correct_predictions / predictions.shape[0]
        avg_accuracy = round(avg_accuracy, 2)
        self._accuracy_label['text'] = str(avg_accuracy) + '%'

    def _show_accuracy_details(self):
        if self._targets is None or self._predictions is None:
            messagebox.showerror('No evaluation Error', 'Evaluation report is missing. Evaluate the matches first.')
            return

        filter_result = self._result_filter_var.get()

        if filter_result == 'All':
            messagebox.showerror('ERROR', 'You need to select odd column first (1/X/2).')
            return
        elif filter_result == '1':
            column = 0
        elif filter_result == 'X':
            column = 1
        elif filter_result == '2':
            column = 2
        else:
            return

        accuracies = self._evaluation_filter.compute_prediction_accuracy_per_odd_range(
            self._inputs[:, column], self._targets, self._predictions
        )
        message_str = ''

        for i in range(self._evaluation_filter.num_intervals):
            message_str += f'Range: {self._evaluation_filter.odd_intervals[i]}, Accuracy: {round(accuracies[i] * 100, 2)}%\n '

        message_str += f'Range > {self._evaluation_filter.end_of_interval}:    Accuracy: {accuracies[-1] * 100}%'
        messagebox.showinfo('Accuracy Details', message_str)


class PredictionDialog(Dialog):
    def __init__(
            self,
            master,
            models: dict,
            results_and_stats: pd.DataFrame
    ):
        self._models = models
        self._results_and_stats = results_and_stats

        self._home_team_var = StringVar()
        self._away_team_var = StringVar()
        self._odd_1_var = StringVar()
        self._odd_x_var = StringVar()
        self._odd_2_var = StringVar()
        self._selected_model_var = StringVar()

        self._prediction_text = None

        super().__init__(master, title='Predictions', window_size={'width': 320, 'height': 450})

    @property
    def models(self):
        return self._models

    @property
    def results_and_stats(self):
        return self._results_and_stats

    def _initialize(self):
        self._initialize_form()

    def _initialize_form(self):
        Label(self._window, text='Home Team', font=('Arial', 12)).place(x=40, y=15)
        Label(self._window, text='Away Team', font=('Arial', 12)).place(x=195, y=15)

        home_team_cb = Combobox(self._window, width=15, font=('Arial', 10), textvariable=self._home_team_var)
        home_team_cb['values'] = self.results_and_stats['Home Team'].unique().tolist()
        home_team_cb.place(x=20, y=50)

        away_team_cb = Combobox(self._window, width=15, font=('Arial', 10), textvariable=self._away_team_var)
        away_team_cb['values'] = self.results_and_stats['Away Team'].unique().tolist()
        away_team_cb.place(x=170, y=50)

        Label(self._window, text='Odd-1', font=('Arial', 12)).place(x=50, y=90)
        Entry(self._window, width=5, font=('Arial', 10), textvariable=self._odd_1_var).place(x=50, y=120)
        Label(self._window, text='Odd-X', font=('Arial', 12)).place(x=135, y=90)
        Entry(self._window, width=5, font=('Arial', 10), textvariable=self._odd_x_var).place(x=135, y=120)
        Label(self._window, text='Odd-2', font=('Arial', 12)).place(x=215, y=90)
        Entry(self._window, width=5, font=('Arial', 10), textvariable=self._odd_2_var).place(x=215, y=120)

        Label(self._window, text='Selected Model', font=('Arial', 12)).place(x=110, y=160)
        models_cb = Combobox(
            self._window, state='readonly', width=20, font=('Arial', 10), textvariable=self._selected_model_var
        )
        models_cb['values'] = list(self._models.keys())
        models_cb.current(0)
        models_cb.place(x=85, y=190)

        Button(self._window, text='Predict', command=self._predict).place(x=130, y=240)

        self._prediction_text = Text(self._window, state=DISABLED, font=('Arial', 10))
        self._prediction_text.place(x=0, y=280)

    def _validate_form(self):
        try:
            float(self._odd_1_var.get())
            float(self._odd_x_var.get())
            float(self._odd_2_var.get())
        except ValueError:
            return 'Odds should be valid positive numbers'

        if float(self._odd_1_var.get()) <= 1.0 or \
                float(self._odd_x_var.get()) <= 1.0 or \
                float(self._odd_2_var.get()) <= 1.0:
            return 'Odds should be greater or equal than 1.0'
        return 'Valid'

    def _predict(self):
        validation_result = self._validate_form()
        if not self._validate_form() == 'Valid':
            messagebox.showerror('Wrong Input', validation_result)
            return

        model = self._models[self._selected_model_var.get()]

        x = preprocessing.construct_input(
            results_and_stats=self.results_and_stats,
            home_team=self._home_team_var.get(),
            away_team=self._away_team_var.get(),
            odd_1=float(self._odd_1_var.get()),
            odd_x=float(self._odd_2_var.get()),
            odd_2=float(self._odd_x_var.get())
        )
        output_proba, y_pred = model.predict(x_inputs=x.reshape((1, -1)))
        self._display_output(output_proba=output_proba[0], y_pred=y_pred[0])

    def _display_output(self, output_proba: np.ndarray, y_pred: int):
        for i, prob in enumerate(output_proba):
            output_proba[i] = round(prob, 2)

        predicted_result_str = {0: 'H', 1: 'X', 2: 'A'}
        text = f'H%: {round(output_proba[0], 2)}\nD%: {round(output_proba[1], 2)}\nA%: {round(output_proba[2], 2)} ' \
               f'\nPredicted Result: {predicted_result_str[y_pred]} '

        self._prediction_text.config(state=NORMAL)
        self._prediction_text.delete('1.0', END)
        self._prediction_text.insert(END, text)
        self._prediction_text.config(state=DISABLED)


class PredictionUpcomingDialog(Dialog):
    def __init__(
            self,
            master,
            models: dict,
            results_and_stats: pd.DataFrame
    ):
        self._models = models
        self._results_and_stats = results_and_stats

        self._fixture_parser = FixtureParser()
        self._similarity_matching = TeamSimilarityMatching()

        self._fixture_teams = None
        self._fixture_odds = None
        self._fixture_inputs = None

        self._treeview_columns = [
            'Home Team', 'Away Team', '1', 'X', '2', 'Prediction', '1 Prob %', 'X Prob %', '2 Prob %'
        ]

        self._fixture_month_var = StringVar()
        self._fixture_day_var = StringVar()
        self._selected_model_var = StringVar()

        self._import_btn = None
        self._predict_btn = None
        self._treeview = None

        super().__init__(master=master, title='Predict Upcoming', window_size={'width': 835, 'height': 780})

    @property
    def models(self) -> dict:
        return self._models

    @property
    def results_and_stats(self) -> pd.DataFrame:
        return self._results_and_stats

    @property
    def fixture_parser(self) -> FixtureParser:
        return self._fixture_parser

    @property
    def similarity_matching(self) -> TeamSimilarityMatching:
        return self._similarity_matching

    @property
    def treeview_columns(self) -> list:
        return self._treeview_columns

    def _initialize(self):
        self._initialize_form()

    def _initialize_form(self):
        self._import_btn = Button(self._window, text='Load Fixtures', command=self._import_fixtures)
        self._import_btn.place(x=40, y=10)

        Label(self._window, text='Load Trained Model', font=('Arial', 12)).place(x=170, y=11)
        models_cb = Combobox(
            self._window, state='readonly', width=25, font=('Arial', 10), textvariable=self._selected_model_var
        )
        models_cb['values'] = list(self._models.keys())
        models_cb.current(0)
        models_cb.place(x=320, y=11)

        Label(self._window, text='Upcoming Date', font=('Arial', 12)).place(x=520, y=11)
        models_cb = Combobox(
            self._window, state='readonly', width=6, font=('Arial', 10), textvariable=self._fixture_month_var
        )
        models_cb['values'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        models_cb.current(0)
        models_cb.place(x=640, y=11)

        models_cb = Combobox(
            self._window, state='readonly', width=6, font=('Arial', 10), textvariable=self._fixture_day_var
        )
        models_cb['values'] = [str(i) for i in range(1, 32)]
        models_cb.current(0)
        models_cb.place(x=720, y=11)

        self._predict_btn = Button(self._window, text='Export Predictions', command=self._predict_fixtures)
        self._predict_btn['state'] = DISABLED
        self._predict_btn.place(x=370, y=710)

        self._treeview = Treeview(
            self._window,
            columns=self.treeview_columns,
            show='headings',
            selectmode='browse',
            height=30
        )
        for column_name in self.treeview_columns:
            self._treeview.column(column_name, anchor=CENTER, stretch=True, width=70)
            self._treeview.heading(column_name, text=column_name, anchor=CENTER)
        self._treeview.column('Home Team', anchor=CENTER, stretch=True, width=100)
        self._treeview.column('Away Team', anchor=CENTER, stretch=True, width=100)
        self._treeview.place(x=50, y=50)

    def _import_fixtures(self):
        fixtures_filepath = filedialog.askopenfilename()

        if '.html' in fixtures_filepath:
            try:
                fixture_matches, fixture_odds = self.fixture_parser.parse_fixture(
                    fixture_filepath=fixtures_filepath,
                    fixtures_month=self._fixture_month_var.get(),
                    fixtures_day=self._fixture_day_var.get()
                )

                if not fixture_matches:
                    messagebox.showerror('Match Parsing ERROR', 'An error occurred while parsing teams.')
                    return
                if not fixture_odds:
                    messagebox.showerror('Odd Parsing ERROR', 'An error occurred while parsing the odds of matches.')
                    return

                self._construct_inputs_from_fixture_matches(
                    fixture_matches=fixture_matches,
                    fixture_odds=fixture_odds
                )
                self._import_btn['state'] = DISABLED
                self._predict_btn['state'] = NORMAL
                messagebox.showinfo('Done', 'Fixtured are imported correctly. Click on Export Predictions.')
            except:
                messagebox.showerror('Parsing ERROR',
                                     'An error occured while trying to parse the fixtures. Check if the upcoming date '
                                     'is correct. If it is, Probably the static HTML code of "footystats.org" has '
                                     'changed and cannot be recognised by the parser. Please, contact the developer.')

    def _construct_inputs_from_fixture_matches(self, fixture_matches: list, fixture_odds: list):
        all_teams = list(
            set(self.results_and_stats['Home Team'].unique().tolist()).union(
                set(self.results_and_stats['Away Team'].unique().tolist())
            ))
        matches = self.similarity_matching.match_teams(
            fixture_matches=fixture_matches,
            all_teams=all_teams
        )

        n_matches = len(matches)
        inputs = []

        for i in range(n_matches):
            home_team, away_team = matches[i]
            odd_1, odd_x, odd_2 = fixture_odds[i]

            x = preprocessing.construct_input(
                results_and_stats=self.results_and_stats,
                home_team=home_team,
                away_team=away_team,
                odd_1=odd_1,
                odd_x=odd_x,
                odd_2=odd_2
            )
            inputs.append(x)
        self._fixture_teams = matches
        self._fixture_odds = fixture_odds
        self._fixture_inputs = np.float64(inputs)

    def _predict_fixtures(self):
        predictions_filepath = filedialog.asksaveasfile(
            filetypes=[('CSV (Excel)', '*.csv')],
            defaultextension='csv'
        ).name

        if predictions_filepath is not None:
            predict_proba, predictions = self._get_predictions()
            predicted_result_dict = {0: 'H', 1: 'X', 2: 'A'}

            rows = [['Home Team', 'Away Team', '1', 'X', '2', 'Prediction', 'Prob(%) 1', 'Prob(%) X', 'Prob(%) 2']]
            for i in range(predict_proba.shape[0]):
                home_team, away_team = self._fixture_teams[i]
                odd_1, odd_x, odd_2 = self._fixture_odds[i]
                y_pred = predicted_result_dict[predictions[i]]
                prob_1, prob_x, prob_2 = predict_proba[i]
                rows.append([home_team, away_team, odd_1, odd_x, odd_2, y_pred, prob_1, prob_x, prob_2])

            self._display_treeview_items(items=rows)
            with open(predictions_filepath, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rows)

    def _get_predictions(self) -> (np.ndarray, np.ndarray):
        model_checkpoint = self._selected_model_var.get()
        model = self._models[model_checkpoint]
        predict_proba, predictions = model.predict(x_inputs=self._fixture_inputs)
        return predict_proba, predictions

    def _display_treeview_items(self, items: list):
        for item in self._treeview.get_children():
            self._treeview.delete(item)

        for i, values in enumerate(items[1:]):
            self._treeview.insert(parent='', index=i, values=values)
