import numpy as np
import pandas as pd
import config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tkinter import StringVar, DoubleVar, messagebox
from tkinter.ttk import Label, Button, Combobox, Treeview, Scrollbar
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.dialog import Dialog
from gui.widgets.percentiles import PercentileSlider
from models.tasks import ClassificationTask
from preprocessing.dataset import DatasetPreprocessor


class EvaluationDialog(Dialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(root=root, title='Model Evaluation', window_size={'width': 950, 'height': 700})

        self._matches_df = matches_df.dropna().reset_index(drop=True)
        self._model_repository = model_repository
        self._league_config = league_config

        self._model_configs = model_repository.get_model_configs(league_id=league_config.league_id)
        self._dataset_preprocessor = DatasetPreprocessor()
        self._tasks = {
            task.name: task for task in [ClassificationTask.Result, ClassificationTask.Over]
            if task.name in self._model_configs and len(self._model_configs[task.name]) > 0
        }
        self._selected_matches_df = None
        self._targets = None
        self._predicted_probabilities = None
        self._predicted_targets = None

        odd_columns = [col for col in ['1', 'X', '2'] if col in self._matches_df.columns]
        self._treeview_columns = {
            ClassificationTask.Result.name: ['Index', 'Date', 'Home Team', 'Away Team'] + odd_columns + ['Result', 'Predicted', 'Prob-H', 'Prob-D', 'Prob-A'],
            ClassificationTask.Over.name: ['Index', 'Date', 'Home Team', 'Away Team'] + odd_columns + ['Result', 'Predicted', 'Prob-U(2.5)', 'Prob-O(2.5)']
        }
        self._odd_filter_values = ['None']
        if '1' in self._matches_df:
            self._odd_filter_values += ['1:(1.00-1.30)', '1:(1.31-1.60)', '1:(1.61-2.00)', '1:(1.00-2.00)', '1:(2.01-3.00)', '1:>3.00']
        if 'X' in self._matches_df:
            self._odd_filter_values += ['X:(1.00-2.00)', 'X:(2.01-3.00)', 'X:(3.01-4.00)', 'X:>4.00']
        if '2' in self._matches_df:
            self._odd_filter_values += ['2:(1.00-1.30)', '2:(1.31-1.60)', '2:(1.61-2.00)', '2:(1.00-2.00)', '2:(2.01-3.00)', '2:>3.00']

        self._home_percent_prob = 0.0
        self._draw_percent_prob = 0.0
        self._away_percent_prob = 0.0
        self._under_percent_prob = 0.0
        self._over_percent_prob = 0.0

        self._samples_values = ['Evaluation', 'Training']
        self._task_var = StringVar()
        self._model_id_var = StringVar()
        self._samples_var = StringVar(value='Evaluation')
        self._odd_filter_var = StringVar()
        self._acc_var = DoubleVar(value=0.0)
        self._f1_var = DoubleVar(value=0.0)
        self._prec_var = DoubleVar(value=0.0)
        self._rec_var = DoubleVar(value=0.0)

        self._treeview = None
        self._treeview_scroll = None
        self._model_cb = None
        self._home_percentile_slider = None
        self._draw_percentile_slider = None
        self._away_percentile_slider = None
        self._under_percentile_slider = None
        self._over_percentile_slider = None
        self._percentiles_btn = None

    def _create_widgets(self):
        Label(self.window, text='Evaluation Samples:', font=('Arial', 11)).place(x=25, y=20)
        samples_cb = Combobox(
            self.window, values=self._samples_values, width=10, font=('Arial', 10), state='readonly', textvariable=self._samples_var
        )
        samples_cb.bind('<<ComboboxSelected>>', self._evaluate)
        samples_cb.place(x=170, y=20)

        Label(self.window, text='Task:', font=('Arial', 11)).place(x=285, y=20)
        task_cb = Combobox(
            self.window, values=list(self._tasks.keys()), width=10, font=('Arial', 10), state='readonly', textvariable=self._task_var
        )
        task_cb.bind('<<ComboboxSelected>>', self._adjust_task)
        task_cb.place(x=340, y=20)

        Label(self.window, text='Model:', font=('Arial', 11)).place(x=455, y=20)
        self._model_cb = Combobox(
            self.window, width=14, font=('Arial', 10), state='readonly', textvariable=self._model_id_var
        )
        self._model_cb.bind('<<ComboboxSelected>>', self._evaluate)
        self._model_cb.place(x=520, y=20)

        Label(self.window, text='Odd Filters:', font=('Arial', 11)).place(x=685, y=20)
        odds_filter_cb = Combobox(
            self.window, values=self._odd_filter_values, width=12, font=('Arial', 10), state='readonly', textvariable=self._odd_filter_var
        )
        odds_filter_cb.bind('<<ComboboxSelected>>', self._display_matches_and_metrics)
        odds_filter_cb.place(x=785, y=20)

        Label(self.window, text='Accuracy:', font=('Arial', 11)).place(x=30, y=635)
        Label(self.window, font=('bold', 11), textvariable=self._acc_var).place(x=100, y=635)

        Label(self.window, text='F1:', font=('Arial', 11)).place(x=150, y=635)
        Label(self.window, font=('bold', 11), textvariable=self._f1_var).place(x=200, y=635)

        Label(self.window, text='Precision:', font=('Arial', 11)).place(x=30, y=665)
        Label(self.window, font=('bold', 11), textvariable=self._prec_var).place(x=100, y=665)

        Label(self.window, text='Recall:', font=('Arial', 11)).place(x=150, y=665)
        Label(self.window, font=('bold', 11), textvariable=self._rec_var).place(x=200, y=665)

        self._percentiles_btn = Button(self.window, text='Store Percent', state='disabled', command=self._store_percentiles_to_model_config)
        self._percentiles_btn.place(x=800, y=640)

    def _reset_metrics(self):
        self._acc_var.set(value=0.0)
        self._f1_var.set(value=0.0)
        self._prec_var.set(value=0.0)
        self._rec_var.set(value=0.0)

    def _adjust_task(self, event):
        def adjust_models(task: str):
            self._model_cb['values'] = list(self._model_configs[task].keys())
            self._model_id_var.set(value='')

        def adjust_treeview(task: str):
            if self._treeview is not None:
                self._treeview.destroy()
                self._treeview_scroll.destroy()

            self._treeview = Treeview(
                self.window,
                show='headings',
                selectmode='extended',
                height=25
            )
            self._treeview.place(x=10, y=60)

            treeview_columns = self._treeview_columns[task]
            self._treeview['columns'] = treeview_columns
            for column_name in treeview_columns:
                self._treeview.column(column_name, anchor='center', stretch=True, width=70)
                self._treeview.heading(column_name, text=column_name, anchor='center')
            self._treeview.column('Home Team', anchor='center', stretch=True, width=100)
            self._treeview.column('Away Team', anchor='center', stretch=True, width=100)
            self._treeview.update()

            self._treeview_scroll = Scrollbar(self.window, orient='vertical', command=self._treeview.yview)
            self._treeview_scroll.place(
                x=self._treeview.winfo_reqwidth() + 30,
                y=60,
                height=self._treeview.winfo_reqheight()
            )
            self._treeview.configure(yscroll=self._treeview_scroll.set)

        task = self._task_var.get()

        if not task:
            return

        self._reset_metrics()
        adjust_models(task=task)
        adjust_treeview(task=task)
        self._percentiles_btn['state'] = 'disabled'

    def _adjust_filters_and_metrics(self, task: str, model_id: str):
        self._odd_filter_var.set(value='None')
        model_config = self._model_configs[task][model_id]

        if self._under_percentile_slider is not None:
            self._under_percentile_slider.destroy()
        if self._over_percentile_slider is not None:
            self._over_percentile_slider.destroy()
        if self._home_percentile_slider is not None:
            self._home_percentile_slider.destroy()
        if self._draw_percentile_slider is not None:
            self._draw_percentile_slider.destroy()
        if self._away_percentile_slider is not None:
            self._away_percentile_slider.destroy()

        if task == 'Result':
            self._home_percentile_slider = PercentileSlider(
                master=self.window, name='Home', initial_value=model_config.home_fixture_percentile[0], x=350, y=615, command=self._display_matches_and_metrics
            )
            self._draw_percentile_slider = PercentileSlider(
                master=self.window, name='Draw', initial_value=model_config.draw_fixture_percentile[0], x=350, y=655, command=self._display_matches_and_metrics
            )
            self._away_percentile_slider = PercentileSlider(
                master=self.window, name='Away', initial_value=model_config.away_fixture_percentile[0], x=560, y=640, command=self._display_matches_and_metrics
            )
        else:
            assert task == 'Over', f'Not defined task: {task}'
            self._under_percentile_slider = PercentileSlider(master=self.window, name='Under(2.5)', initial_value=model_config.over_fixture_percentile[0], x=400, y=615)
            self._over_percentile_slider = PercentileSlider(master=self.window, name='Over(2.5)', initial_value=model_config.under_fixture_percentile[0], x=400, y=655)

        self._percentiles_btn['state'] = 'normal'

    def _store_percentiles_to_model_config(self):
        task = self._task_var.get()
        model_id = self._model_id_var.get()
        model_config = self._model_configs[task][model_id]

        if task == 'Result':
            model_config.home_fixture_percentile = (self._home_percentile_slider.get_value(), self._home_percent_prob)
            model_config.draw_fixture_percentile = (self._draw_percentile_slider.get_value(), self._draw_percent_prob)
            model_config.away_fixture_percentile = (self._away_percentile_slider.get_value(), self._away_percent_prob)
        else:
            assert task == 'Over', f'Not defined task: {task}'

            model_config.under_fixture_percentile = (self._under_percentile_slider.get_value(), self._under_percent_prob)
            model_config.over_fixture_percentile = (self._over_percentile_slider.get_value(), self._over_percent_prob)

        self._model_repository.update_model_config(model_config=model_config)

    def _clear_matches(self):
        for item in self._treeview.get_children():
            self._treeview.delete(item)

    def _add_matches(self, filter_ids: np.ndarray):
        match_columns = ['Date', 'Home Team', 'Away Team'] + [col for col in ['1', 'X', '2'] if col in self._matches_df.columns]
        matches_df = self._selected_matches_df[filter_ids][match_columns]

        if matches_df.shape[0] > 0:
            targets = self._targets[filter_ids]
            predictions = self._predicted_targets[filter_ids]
            predicted_probabilities = self._predicted_probabilities[filter_ids]

            matches_df.insert(loc=0, column='Index', value=np.arange(1, matches_df.shape[0] + 1))
            matches_df['Result'] = targets
            matches_df['Predicted'] = predictions

            task = self._task_var.get()

            if task == 'Result':
                assert self._predicted_probabilities.shape[1] == 3, 'Incorrect probabilities passed into _add_matches function'

                matches_df['Result'] = matches_df['Result'].replace({0: 'H', 1: 'D', 2: 'A'})
                matches_df['Predicted'] = matches_df['Predicted'].replace({0: 'H', 1: 'D', 2: 'A'})
                matches_df['Prob-H'] = predicted_probabilities[:, 0]
                matches_df['Prob-D'] = predicted_probabilities[:, 1]
                matches_df['Prob-A'] = predicted_probabilities[:, 2]
            elif task == 'Over':
                assert self._predicted_probabilities.shape[1] == 2, 'Incorrect probabilities passed into _add_matches function'

                matches_df['Result'] = matches_df['Result'].replace({0: 'U(2.5)', 1: 'O(2.5)'})
                matches_df['Predicted'] = matches_df['Predicted'].replace({0: 'U(2.5)', 1: 'O(2.5)'})
                matches_df['Prob-U(2.5)'] = predicted_probabilities[:, 0]
                matches_df['ProbO(2.5)'] = predicted_probabilities[:, 1]
            else:
                raise NotImplementedError(f'Not implemented task: {task}')

            for i, values in enumerate(matches_df.values.tolist()):
                self._treeview.insert(parent='', index=i, values=values)

            correct_ids = targets == predictions
            self._highlight_correct_matches(correct_ids=correct_ids)

    def _highlight_correct_matches(self, correct_ids: np.ndarray):
        previously_selected_items = self._treeview.selection()

        if len(previously_selected_items) > 0:
            self._treeview.selection_remove(previously_selected_items)

        items = self._treeview.get_children()
        selections = [item for item, is_correct in zip(items, correct_ids) if is_correct]
        self._treeview.selection_set(selections)

    def _display_matches_and_metrics(self, event=None):
        def get_filtered_mask() -> np.ndarray:
            filter_str = self._odd_filter_var.get()

            if filter_str == 'None':
                return np.array([True]*self._selected_matches_df.shape[0])
            else:
                odd, cond = filter_str.split(':')
                if cond[0] == '>':
                    return self._selected_matches_df[odd] > float(cond[1:])
                else:
                    cond_min, cond_max = cond[1: -1].split('-')
                    return (self._selected_matches_df[odd] >= float(cond_min)) & (self._selected_matches_df[odd] <= float(cond_max))

        def get_percentile_mask() -> np.ndarray:
            def get_mask(correct_ids: np.bool, target_col: int, percentile: int) -> np.ndarray or bool:
                correct_proba = self._predicted_probabilities[correct_ids & (self._targets == target_col), target_col]

                if correct_proba.shape[0] == 0:
                    return False, 0.0
                elif percentile == 0:
                    return self._predicted_targets == target_col, 0.0
                else:
                    calib_prob = np.percentile(a=correct_proba, q=percentile)
                    return self._predicted_probabilities[:, target_col] >= calib_prob, calib_prob

            correct_ids = self._targets == self._predicted_targets
            task = self._task_var.get()

            if task == 'Result':
                home_mask, prob = get_mask(correct_ids=correct_ids, target_col=0, percentile=self._home_percentile_slider.get_value())
                self._home_percent_prob = prob
                draw_mask, prob = get_mask(correct_ids=correct_ids, target_col=1, percentile=self._draw_percentile_slider.get_value())
                self._draw_percent_prob = prob
                away_mask, prob = get_mask(correct_ids=correct_ids, target_col=2, percentile=self._away_percentile_slider.get_value())
                self._away_percent_prob = prob
                return (home_mask | draw_mask) | away_mask
            elif task == 'Over':
                under_mask, prob = get_mask(correct_ids=correct_ids, target_col=0, percentile=self._under_percentile_slider.get_value())
                self._under_percent_prob = prob
                over_mask, prob = get_mask(correct_ids=correct_ids, target_col=1, percentile=self._over_percentile_slider.get_value())
                self._over_percent_prob = prob
                return under_mask | over_mask
            else:
                raise NotImplementedError(f'Undefined task: "{task}"')

        if self._targets is None:
            return

        self._clear_matches()
        odd_ids = get_filtered_mask()
        percentile_ids = get_percentile_mask()
        filter_ids = (odd_ids & percentile_ids)
        self._compute_metrics(filter_ids=filter_ids)
        self._add_matches(filter_ids=filter_ids)

    def _evaluate(self, event):
        model_id = self._model_id_var.get()

        if not model_id:
            return

        task = self._task_var.get()
        self._adjust_filters_and_metrics(task=task, model_id=model_id)

        samples_str = self._samples_var.get()
        if samples_str == 'Evaluation':
            self._selected_matches_df = self._matches_df[: config.fit_test_size]
        else:
            assert samples_str == 'Training', f'Undefined samples: {samples_str}'

            self._selected_matches_df = self._matches_df[config.fit_test_size:]

        model_config = self._model_configs[task][model_id]
        model = self._model_repository.load_model(model_config=model_config)

        x, self._targets, normalizer, _ = self._dataset_preprocessor.preprocess_dataset(
            df=self._selected_matches_df,
            task=self._tasks[task],
            fit_normalizer=False,
            normalizer=model_config.normalizer,
            sampler=None
        )
        self._predicted_probabilities = np.round(model.predict_proba(x=x), decimals=2)
        self._predicted_targets = self._predicted_probabilities.argmax(axis=1)
        self._display_matches_and_metrics()

    def _compute_metrics(self, filter_ids: np.ndarray):
        if filter_ids.shape[0] == 0:
            self._reset_metrics()
        else:
            task = self._task_var.get()

            if task == 'Result':
                average = 'macro'
            else:
                assert task == 'Over', f'Undefined task: {task}'

                average = 'binary'

            targets = self._targets[filter_ids]
            predictions = self._predicted_targets[filter_ids]
            self._acc_var.set(value=round(accuracy_score(y_true=targets, y_pred=predictions), 2))
            self._f1_var.set(value=round(f1_score(y_true=targets, y_pred=predictions, average=average, zero_division=0.0), 2))
            self._prec_var.set(value=round(precision_score(y_true=targets, y_pred=predictions, average=average, zero_division=0.0), 2))
            self._rec_var.set(value=round(recall_score(y_true=targets, y_pred=predictions, average=average, zero_division=0.0), 2))

    def _init_dialog(self):
        messagebox.showinfo(
            parent=self.window,
            title='Filters',
            message='Use odd filters to select rows (matches) that satisfy a filter condition. (e.g. all matches where home prob 1 > 3.10)'
                    '\nUse percentiles to filter minimum predicted probabilities to be displayed for a given target (e.g. all home prob > 5% of all home probs)'
                    '\nFor more info, refer to help menu.'
        )

    def _get_dialog_result(self):
        return None
