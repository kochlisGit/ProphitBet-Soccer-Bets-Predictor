from database import utils
from windows.dialogs.dialog import Dialog
from windows.dialogs.task import TaskLoaderDialog
from tkinter import messagebox, Text, StringVar, DISABLED, NORMAL, END, CENTER, VERTICAL, HORIZONTAL
from tkinter.ttk import Label, Entry, Button, Treeview, Combobox, Scrollbar
from models.model import FCModel
from database.utils import preprocess_data
from analysis.predictions.filters import EvaluationFilter
from itertools import compress
import numpy as np
import threading


class TrainDialog(Dialog):
    def __init__(self, master, directory_path, results_and_stats, columns):
        self._model_name = directory_path
        self._results_and_stats = results_and_stats
        self._columns = columns

        self._default_hidden_layers = '32, 64, 64, 32'
        self._default_validation_size = '100'
        self._default_noise = 0.15
        self._default_epochs = 150
        self._default_patience = 50

        self._title = 'Training'
        self._window_sizes = {'width': 400, 'height': 600}

        self._layers_var = StringVar()
        self._validation_var = StringVar()
        self._noise_var = StringVar()
        self._epochs_var = StringVar()
        self._patience_var = StringVar()

        self._text_area = None
        self._training_on_progress = False

        super().__init__(master)

    def _initialize(self):
        self._initialize_window()
        self._initialize_form()

    def _initialize_window(self):
        self._window.title(self._title)
        self._window.geometry('{}x{}'.format(self._window_sizes['width'], self._window_sizes['height']))
        self._window.resizable(False, False)

    def _initialize_form(self):
        Label(self._window, text='Model\'s Name', font=('Arial', 16)).place(x=30, y=20)
        Label(self._window, text='Hidden Layers', font=('Arial', 16)).place(x=30, y=60)
        Label(self._window, text='Validation Size', font=('Arial', 16)).place(x=30, y=100)
        Label(self._window, text='Noise Range', font=('Arial', 16)).place(x=30, y=140)
        Label(self._window, text='Epochs', font=('Arial', 16)).place(x=30, y=180)
        Label(self._window, text='Patience', font=('Arial', 16)).place(x=30, y=220)

        name_entry = Entry(self._window, width=15, font=('Arial', 10))
        name_entry.insert(0, self._model_name)
        name_entry.config(state=DISABLED)
        name_entry.place(x=200, y=20)

        layers_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._layers_var)
        layers_entry.insert(0, self._default_hidden_layers)
        layers_entry.place(x=200, y=60)

        validation_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._validation_var)
        validation_entry.insert(0, self._default_validation_size)
        validation_entry.place(x=200, y=100)

        noise_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._noise_var)
        noise_entry.insert(0, str(self._default_noise))
        noise_entry.place(x=200, y=140)

        epochs_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._epochs_var)
        epochs_entry.insert(0, str(self._default_epochs))
        epochs_entry.place(x=200, y=180)

        patience_entry = Entry(self._window, width=15, font=('Arial', 10), textvariable=self._patience_var)
        patience_entry.insert(0, str(self._default_patience))
        patience_entry.place(x=200, y=220)

        Button(self._window, text='Train', command=self._train).place(x=150, y=260)

        self._text_area = Text(self._window, state=DISABLED, font=('Arial', 10))
        self._text_area.place(x=0, y=300)

    def _report_training_log(self, text):
        self._text_area.config(state=NORMAL)
        self._text_area.insert(END, text)
        self._text_area.config(state=DISABLED)

    def _clear_training_log(self):
        self._text_area.config(state=NORMAL)
        self._text_area.delete('1.0', END)
        self._text_area.config(state=DISABLED)

    def _validate_form(self):
        layers = self._layers_var.get()
        layers = layers.replace(' ', '')
        layers_list = layers.split(',')

        validation_size = self._validation_var.get()

        noise = self._noise_var.get()
        noise = noise.replace('.', '')

        epochs = self._epochs_var.get()
        patience = self._patience_var.get()

        for n_layers in layers_list:
            if not n_layers.isdigit():
                return 'Layers should be positive numbers, separated by ","'
        if not validation_size.isdigit() or int(validation_size) < 0:
            return 'Validation Size should be a positive number'
        if not noise.isdigit() or float(noise) < 0:
            return 'Noise Range should have a positive value < 1.0'
        if not epochs.isdigit() or int(epochs) < 0:
            return 'Epochs should be a positive number'
        if not patience.isdigit() or int(patience) < 0:
            return 'Patience should be a positive number'
        return 'Valid'

    def _train(self):
        form_validation_result = self._validate_form()

        if form_validation_result == 'Valid':
            if self._training_on_progress:
                messagebox.showerror('ERROR', 'Another training is on progress.')
            else:
                train_thread = threading.Thread(target=self._train_model)
                train_thread.start()
        else:
            messagebox.showerror('ERROR', form_validation_result)

    def _train_model(self):
        self._training_on_progress = True
        default_train_log = 'Current Epoch: {}, Validation Accuracy = {}\n'

        layers = self._layers_var.get()
        layers = layers.replace(' ', '')
        layers_list = layers.split(',')
        hidden_layers = [int(units) for units in layers_list]

        validation_size = int(self._validation_var.get())
        noise_range = float(self._noise_var.get())
        epochs = int(self._epochs_var.get())
        patience = int(self._patience_var.get())

        self._report_training_log('Preprocessing data...\n')

        inputs, targets = preprocess_data(self._results_and_stats, self._columns)

        x_test = inputs[:validation_size]
        y_test = targets[:validation_size]
        x_train = inputs[validation_size:]
        y_train = targets[validation_size:]

        self._report_training_log('Building the model...\n')

        model = FCModel(self._model_name)
        model.build_model(inputs.shape[1:], hidden_layers)

        current_epoch = 0
        best_accuracy = 0
        patience_counter = 0

        self._report_training_log('Ready to train...\n')

        while current_epoch < epochs and patience_counter < patience:
            accuracy = round(model.train(x_train, y_train, x_test, y_test, noise_range), 2)

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


class EvaluationDialog(Dialog):
    def __init__(self, master, directory_path, results_and_stats, all_columns, basic_columns):
        self._model_name = directory_path
        self._results_and_stats = results_and_stats
        self._all_columns = all_columns
        self._basic_columns = basic_columns
        self._evaluation_columns = [
            'Prediction',
            '1 Prob %',
            'X Prob %',
            '2 Prob %'
        ]
        self._last_n_values = [10, 25, 50, 100, 150, 200]
        self._result_filter_values = ['All', '1', 'X', '2']
        self._inputs = None
        self._targets = None
        self._evaluation_results_and_stats = None
        self._predicted_results = None
        self._evaluation_filter = EvaluationFilter()

        self._title = 'Evaluation'
        self._loading_title = 'Loading Model'
        self._window_sizes = {'width': 1050, 'height': 700}

        self._last_n_var = StringVar()
        self._result_filter_var = StringVar()

        self._treeview_columns = basic_columns + self._evaluation_columns
        self._treeview = None
        self._accuracy_label = None
        self._model = None

        super().__init__(master)

    def _initialize(self):
        self._initialize_window()
        self._initialize_model()
        self._initialize_form()

    def _initialize_window(self):
        self._window.title(self._title)
        self._window.geometry('{}x{}'.format(self._window_sizes['width'], self._window_sizes['height']))
        self._window.resizable(False, False)

    def _initialize_form(self):
        Label(self._window, text='Last N:', font=('Arial', 12)).place(x=50, y=10)

        leagues_cb = Combobox(
            self._window, state='readonly', width=10, font=('Arial', 10), textvariable=self._last_n_var
        )
        leagues_cb['values'] = self._last_n_values
        leagues_cb.place(x=120, y=10)
        leagues_cb.bind('<<ComboboxSelected>>', self._eval_matches)

        Label(self._window, text='Odds Filter:', font=('Arial', 12)).place(x=250, y=10)

        result_filter_cb = Combobox(
            self._window, state='readonly', width=10, font=('Arial', 10), textvariable=self._result_filter_var
        )
        result_filter_cb['values'] = self._result_filter_values
        result_filter_cb.current(0)
        result_filter_cb.place(x=350, y=10)
        result_filter_cb.bind('<<ComboboxSelected>>', self._eval_matches)

        Button(self._window, text='Show Filtered Accuracy', command=self._display_detailed_accuracy).place(x=500, y=10)

        Label(self._window, text='Accuracy:', font=('Arial', 12)).place(x=700, y=10)
        self._accuracy_label = Label(self._window, font=('Arial', 10))
        self._accuracy_label.place(x=770, y=13)

        self._treeview = Treeview(
            self._window,
            columns=self._treeview_columns,
            show='headings',
            selectmode='browse',
            height=30
        )
        for column_name in self._treeview_columns:
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

    def _initialize_model(self):
        task_dialog = TaskLoaderDialog(self._window, self._loading_title)
        load_model_thread = threading.Thread(target=self._load_model, args=(task_dialog,))
        load_model_thread.start()
        task_dialog.start()
        load_model_thread.join()

    def _load_model(self, task_dialog: TaskLoaderDialog):
        self._model = FCModel(self._model_name)
        self._model.load()
        task_dialog.exit()

    def _filter_data(self, inputs, targets, evaluation_results_and_stats):
        filter_result = self._result_filter_var.get()

        if filter_result == 'All':
            return inputs, targets, evaluation_results_and_stats
        elif filter_result == '1':
            indices = targets == 0
        elif filter_result == 'X':
            indices = targets == 1
        elif filter_result == '2':
            indices = targets == 2
        else:
            return inputs, targets, evaluation_results_and_stats
        return inputs[indices], targets[indices], list(compress(evaluation_results_and_stats, indices))

    def _eval_matches(self, event):
        last_n_var = self._last_n_var.get()

        if last_n_var == '':
            return

        last_n = int(last_n_var)
        evaluation_results_and_stats = self._results_and_stats[:last_n]
        inputs, targets = preprocess_data(evaluation_results_and_stats, self._all_columns)
        targets = np.argmax(targets, axis=1)
        self._inputs, self._targets, self._evaluation_results_and_stats = self._filter_data(
            inputs,
            targets,
            evaluation_results_and_stats
        )

        predictions = self._model.predict(self._inputs)
        for i in range(predictions.shape[0]):
            for result_index in range(3):
                predictions[i][result_index] = round(predictions[i][result_index], 2)
        self._predicted_results = np.argmax(predictions, axis=1)

        self._display_treeview_items(self._evaluation_results_and_stats, predictions, self._predicted_results)
        self._display_validation_accuracy(self._targets, self._predicted_results)

    def _display_treeview_items(self, evaluation_results_and_stats, predictions, predicted_results):
        for item in self._treeview.get_children():
            self._treeview.delete(item)

        predicted_results = utils.predictions_to_result(predicted_results)
        items = []
        n_basic_columns = len(self._basic_columns)
        for i in range(predictions.shape[0]):
            basic_data = evaluation_results_and_stats[i][:n_basic_columns]
            model_predicted_result = [predicted_results[i]]
            model_predictions = [predictions[i][0], predictions[i][1], predictions[i][2]]
            items.append(basic_data + model_predicted_result + model_predictions)

        for i, values in enumerate(items):
            self._treeview.insert(parent='', index=i, values=values)

    def _display_validation_accuracy(self, y_targets, predicted_results):
        correct_predictions = 0
        n_predictions = predicted_results.shape[0]

        for i in range(n_predictions):
            if y_targets[i] == predicted_results[i]:
                correct_predictions += 1
        accuracy = round(100 * correct_predictions/n_predictions, 2)

        self._accuracy_label['text'] = str(accuracy) + '%'

    def _display_detailed_accuracy(self):
        if self._targets is None:
            return

        filter_result = self._result_filter_var.get()

        if filter_result == 'All':
            messagebox.showerror('ERROR', 'You need to select result first.')
            return
        elif filter_result == '1':
            column = 0
        elif filter_result == 'X':
            column = 1
        elif filter_result == '2':
            column = 2
        else:
            return

        accuracies = self._evaluation_filter.filter_odd_accuracy_per_range(
            self._inputs, self._targets, self._predicted_results, column
        )
        message_str = ''

        for i in range(self._evaluation_filter.n_ranges):
            message_str += 'Range: {}, Accuracy: {}%\n'.format(
                self._evaluation_filter.odd_ranges[i], round(accuracies[i], 2)
            )
        message_str += 'Range > {}, Accuracy: {}%'.format(
            self._evaluation_filter.end_of_range,
            accuracies[self._evaluation_filter.n_ranges]
        )
        messagebox.showinfo('Detailed Accuracies', message_str)


class PredictionsDialog(Dialog):
    def __init__(self, master, directory_path, results_and_stats, columns):
        self._model_name = directory_path
        self._results_and_stats = results_and_stats
        self._columns = columns

        self._title = 'Predictions'
        self._loading_title = 'Loading Model'
        self._window_sizes = {'width': 320, 'height': 400}

        self._home_team_var = StringVar()
        self._away_team_var = StringVar()
        self._odd_1_var = StringVar()
        self._odd_x_var = StringVar()
        self._odd_2_var = StringVar()
        self._prediction_text = None

        self._model = None

        super().__init__(master)

    def _initialize(self):
        self._initialize_window()
        self._initialize_model()
        self._initialize_form()

    def _initialize_window(self):
        self._window.title(self._title)
        self._window.geometry('{}x{}'.format(self._window_sizes['width'], self._window_sizes['height']))
        self._window.resizable(False, False)

    def _initialize_form(self):
        Label(self._window, text='Home Team', font=('Arial', 12)).place(x=40, y=15)
        Label(self._window, text='Away Team', font=('Arial', 12)).place(x=195, y=15)

        all_teams = list(utils.get_all_league_teams(self._results_and_stats))
        all_teams.sort()

        home_team_cb = Combobox(self._window, width=15, font=('Arial', 10), textvariable=self._home_team_var)
        home_team_cb['values'] = all_teams
        home_team_cb.place(x=20, y=50)

        away_team_cb = Combobox(self._window, width=15, font=('Arial', 10), textvariable=self._away_team_var)
        away_team_cb['values'] = all_teams
        away_team_cb.place(x=170, y=50)

        Label(self._window, text='Odd-1', font=('Arial', 12)).place(x=50, y=90)
        Entry(self._window, width=5, font=('Arial', 10), textvariable=self._odd_1_var).place(x=50, y=120)
        Label(self._window, text='Odd-X', font=('Arial', 12)).place(x=135, y=90)
        Entry(self._window, width=5, font=('Arial', 10), textvariable=self._odd_x_var).place(x=135, y=120)
        Label(self._window, text='Odd-2', font=('Arial', 12)).place(x=215, y=90)
        Entry(self._window, width=5, font=('Arial', 10), textvariable=self._odd_2_var).place(x=215, y=120)

        Button(self._window, text='Predict', command=self._predict).place(x=130, y=160)

        self._prediction_text = Text(self._window, state=DISABLED, font=('Arial', 10))
        self._prediction_text.place(x=0, y=200)

    def _initialize_model(self):
        task_dialog = TaskLoaderDialog(self._window, self._loading_title)
        load_model_thread = threading.Thread(target=self._load_model, args=(task_dialog,))
        load_model_thread.start()
        task_dialog.start()
        load_model_thread.join()

    def _load_model(self, task_dialog: TaskLoaderDialog):
        self._model = FCModel(self._model_name)
        self._model.load()
        task_dialog.exit()

    def _display_predictions(self, result_probabilities, predicted_result):
        predicted_result = utils.predictions_to_result([predicted_result])[0]
        prob_1 = round(result_probabilities[0], 2)
        prob_x = round(result_probabilities[1], 2)
        prob_2 = round(result_probabilities[2], 2)

        text = 'H%: {:.2f}\nD%: {:.2f}\nA%: {:.2f}\nPredicted Result: {}'.format(
            prob_1,
            prob_x,
            prob_2,
            predicted_result
        )

        self._prediction_text.config(state=NORMAL)
        self._prediction_text.delete('1.0', END)
        self._prediction_text.insert(END, text)
        self._prediction_text.config(state=DISABLED)

    def _validate_form(self):
        odd_1 = self._odd_1_var.get().replace('.', '')
        if not odd_1.isdigit() and float(odd_1) > 0:
            return 'Odd 1 is not a valid positive number'
        odd_x = self._odd_x_var.get().replace('.', '')
        if not odd_x.isdigit() and float(odd_x) > 0:
            return 'Odd X is not a valid positive number'
        odd_2 = self._odd_2_var.get().replace('.', '')
        if not odd_2.isdigit() and float(odd_2) > 0:
            return 'Odd 2 is not a valid positive number'
        return 'valid'

    def _predict(self):
        home_team = self._home_team_var.get()
        away_team = self._away_team_var.get()
        odd_1 = float(self._odd_1_var.get())
        odd_x = float(self._odd_x_var.get())
        odd_2 = float(self._odd_2_var.get())

        validation_result = self._validate_form()
        if validation_result == 'valid':
            pass
        else:
            messagebox.showerror('showerror', 'ERROR: ' + validation_result)

        x = utils.construct_prediction_sample(
            self._results_and_stats,
            self._columns,
            home_team,
            away_team,
            odd_1,
            odd_x,
            odd_2
        )
        result_probabilities = self._model.predict(x)[0]
        predicted_result = np.argmax(result_probabilities)
        self._display_predictions(result_probabilities, predicted_result)
