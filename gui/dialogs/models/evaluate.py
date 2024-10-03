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

        self._all_matches_df = matches_df.dropna().reset_index(drop=True)
        self._model_repository = model_repository

        self._model_configs = model_repository.get_model_configs(league_id=league_config.league_id)
        self._dataset_preprocessor = DatasetPreprocessor()
        self._tasks = {
            task.name: task for task in [ClassificationTask.Result, ClassificationTask.Over]
            if task.name in self._model_configs and len(self._model_configs[task.name]) > 0
        }
        self._matches_df = None
        self._targets = None
        self._predicted_probabilities = None
        self._predicted_targets = None
        self._samples_values = ['Evaluation', 'Training']
        self._home_percent_prob_filter = 0.0
        self._draw_percent_prob_filter = 0.0
        self._away_percent_prob_filter = 0.0
        self._under_percent_prob_filter = 0.0
        self._over_percent_prob_filter = 0.0

        odd_columns = [col for col in ['1', 'X', '2'] if col in self._all_matches_df.columns]
        self._treeview_columns = {
            ClassificationTask.Result.name: ['Index', 'Date', 'Home Team', 'Away Team'] + odd_columns + ['Result', 'Predicted', 'Prob-H', 'Prob-D', 'Prob-A'],
            ClassificationTask.Over.name: ['Index', 'Date', 'Home Team', 'Away Team'] + odd_columns + ['Result', 'Predicted', 'Prob-U(2.5)', 'Prob-O(2.5)']
        }
        self._odd_filter_values = ['None']
        if '1' in self._all_matches_df:
            self._odd_filter_values += ['1:(1.00-1.30)', '1:(1.31-1.60)', '1:(1.61-2.00)', '1:(1.00-2.00)', '1:(2.01-3.00)', '1:>3.00']
        if 'X' in self._all_matches_df:
            self._odd_filter_values += ['X:(1.00-2.00)', 'X:(2.01-3.00)', 'X:(3.01-4.00)', 'X:>4.00']
        if '2' in self._all_matches_df:
            self._odd_filter_values += ['2:(1.00-1.30)', '2:(1.31-1.60)', '2:(1.61-2.00)', '2:(1.00-2.00)', '2:(2.01-3.00)', '2:>3.00']

        self._task_var = StringVar()
        self._model_id_var = StringVar()
        self._samples_var = StringVar(value='')
        self._odd_filter_var = StringVar(value='')
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
        self._store_filters_btn = None

    def _create_widgets(self):
        Label(self.window, text='Evaluation Samples:', font=('Arial', 11)).place(x=25, y=20)
        samples_cb = Combobox(
            self.window, values=self._samples_values, width=10, font=('Arial', 10), state='readonly', textvariable=self._samples_var
        )
        samples_cb.bind('<<ComboboxSelected>>', self._on_samples_select)
        samples_cb.place(x=170, y=20)

        Label(self.window, text='Task:', font=('Arial', 11)).place(x=285, y=20)
        task_cb = Combobox(
            self.window, values=list(self._tasks.keys()), width=10, font=('Arial', 10), state='readonly', textvariable=self._task_var
        )
        task_cb.bind('<<ComboboxSelected>>', self._on_task_select)
        task_cb.place(x=340, y=20)

        Label(self.window, text='Model:', font=('Arial', 11)).place(x=455, y=20)
        self._model_cb = Combobox(
            self.window, width=14, font=('Arial', 10), state='readonly', textvariable=self._model_id_var
        )
        self._model_cb.bind('<<ComboboxSelected>>', self._on_model_select)
        self._model_cb.place(x=520, y=20)

        Label(self.window, text='Odd Filters:', font=('Arial', 11)).place(x=685, y=20)
        odds_filter_cb = Combobox(
            self.window, values=self._odd_filter_values, width=12, font=('Arial', 10), state='readonly', textvariable=self._odd_filter_var
        )
        odds_filter_cb.bind('<<ComboboxSelected>>', self._on_filter_select)
        odds_filter_cb.place(x=785, y=20)

        Label(self.window, text='Accuracy:', font=('Arial', 11)).place(x=30, y=635)
        Label(self.window, font=('bold', 11), textvariable=self._acc_var).place(x=100, y=635)
        Label(self.window, text='F1:', font=('Arial', 11)).place(x=150, y=635)
        Label(self.window, font=('bold', 11), textvariable=self._f1_var).place(x=200, y=635)
        Label(self.window, text='Precision:', font=('Arial', 11)).place(x=30, y=665)
        Label(self.window, font=('bold', 11), textvariable=self._prec_var).place(x=100, y=665)
        Label(self.window, text='Recall:', font=('Arial', 11)).place(x=150, y=665)
        Label(self.window, font=('bold', 11), textvariable=self._rec_var).place(x=200, y=665)

        self._store_filters_btn = Button(self.window, text='Store Filters', state='disabled', command=self._store_filters)
        self._store_filters_btn.place(x=800, y=660)
        self._store_filters_btn['state'] = 'disabled'

    def _on_samples_select(self, event):
        samples_str = self._samples_var.get()

        if samples_str == 'Evaluation':
            self._matches_df = self._all_matches_df[: config.fit_test_size]
        else:
            assert samples_str == 'Training', f'Undefined samples: {samples_str}'

            self._matches_df = self._all_matches_df[config.fit_test_size:]

        self._evaluate(event=event)

    def _on_task_select(self, event):
        def reset_metrics():
            self._acc_var.set(value=0.0)
            self._f1_var.set(value=0.0)
            self._prec_var.set(value=0.0)
            self._rec_var.set(value=0.0)

        def reset_models(task: str):
            self._model_cb['values'] = list(self._model_configs[task].keys())
            self._model_id_var.set(value='')

        def reset_filters(task: str):
            def delete_percentile_sliders():
                if self._home_percentile_slider is not None:
                    self._home_percentile_slider.destroy()
                    self._home_percentile_slider = None
                if self._draw_percentile_slider is not None:
                    self._draw_percentile_slider.destroy()
                    self._draw_percentile_slider = None
                if self._away_percentile_slider is not None:
                    self._away_percentile_slider.destroy()
                    self._away_percentile_slider = None
                if self._under_percentile_slider is not None:
                    self._under_percentile_slider.destroy()
                    self._under_percentile_slider = None
                if self._over_percentile_slider is not None:
                    self._over_percentile_slider.destroy()
                    self._over_percentile_slider = None

            delete_percentile_sliders()
            self._odd_filter_var.set(value='None')
            self._load_percentile_filters(task=task)
            self._store_filters_btn['state'] = 'normal'

        task = self._task_var.get()

        if not task:
            return

        reset_metrics()
        reset_models(task=task)
        reset_filters(task=task)
        self._reset_treeview(task=task)

    def _on_model_select(self, event):
        self._odd_filter_var.set(value='None')
        self._load_percentile_filters(task=self._task_var.get())
        self._evaluate(event=event)

    def _on_filter_select(self, event):
        self._load_percentile_filters(task=self._task_var.get())
        self._display_matches_and_metrics()

    def _store_filters(self):
        model_id = self._model_id_var.get()

        if not model_id:
            return

        task = self._task_var.get()

        if task == 'Result':
            percentiles = {
                'home': (self._home_percentile_slider.get_value(), self._home_percent_prob_filter),
                'draw': (self._draw_percentile_slider.get_value(), self._draw_percent_prob_filter),
                'away': (self._away_percentile_slider.get_value(), self._away_percent_prob_filter)
            }
        else:
            percentiles = {
                'under': (self._under_percentile_slider.get_value(), self._under_percent_prob_filter),
                'over': (self._over_percentile_slider.get_value(), self._over_percent_prob_filter)
            }

        model_config = self._model_configs[task][model_id]
        model_config.odds_filter[self._odd_filter_var.get()] = percentiles
        self._model_repository.update_model_config(model_config=model_config)

    def _evaluate(self, event):
        if self._matches_df is None:
            return

        model_id = self._model_id_var.get()

        if not model_id:
            return

        task = self._task_var.get()

        if not task:
            return

        model_config = self._model_configs[task][model_id]
        model = self._model_repository.load_model(model_config=model_config)

        x, self._targets, normalizer, _ = self._dataset_preprocessor.preprocess_dataset(
            df=self._matches_df,
            task=self._tasks[task],
            fit_normalizer=False,
            normalizer=model_config.normalizer,
            sampler=None
        )
        self._predicted_probabilities = np.round(model.predict_proba(x=x), decimals=2)
        self._predicted_targets = self._predicted_probabilities.argmax(axis=1)
        self._display_matches_and_metrics()

    def _reset_treeview(self, task: str):
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

    def _load_percentile_filters(self, task: str):
        percentiles = {'home': (0, 0), 'draw': (0, 0), 'away': (0, 0), 'under': (0, 0), 'over': (0, 0)}
        model_id = self._model_id_var.get()

        if model_id:
            model_config = self._model_configs[task][model_id]
            filter_id = self._odd_filter_var.get()

            if filter_id in model_config.odds_filter:
                percentiles = model_config.odds_filter[filter_id]

        if task == 'Result':
            self._home_percentile_slider = PercentileSlider(
                master=self.window, name='Home', initial_value=percentiles['home'][0], x=350, y=615, command=self._display_matches_and_metrics
            )
            self._home_percent_prob_filter = percentiles['home'][1]
            self._draw_percentile_slider = PercentileSlider(
                master=self.window, name='Draw', initial_value=percentiles['draw'][0], x=350, y=655, command=self._display_matches_and_metrics
            )
            self._draw_percent_prob_filter = percentiles['draw'][1]
            self._away_percentile_slider = PercentileSlider(
                master=self.window, name='Away', initial_value=percentiles['away'][0], x=560, y=640, command=self._display_matches_and_metrics
            )
            self._away_percent_prob_filter = percentiles['away'][1]
        else:
            assert task == 'Over', f'Not defined task: {task}'

            self._under_percentile_slider = PercentileSlider(
                master=self.window, name='Under(2.5)', initial_value=percentiles['under'][0], x=400, y=615, command=self._display_matches_and_metrics
            )
            self._under_percent_prob_filter = percentiles['under'][1]
            self._over_percentile_slider = PercentileSlider(
                master=self.window, name='Over(2.5)', initial_value=percentiles['over'][0], x=400, y=655, command=self._display_matches_and_metrics
            )
            self._over_percent_prob_filter = percentiles['over'][1]

    def _display_matches_and_metrics(self):
        def clear_treeview_items():
            for item in self._treeview.get_children():
                self._treeview.delete(item)

        def get_odd_mask() -> np.ndarray:
            filter_str = self._odd_filter_var.get()

            if filter_str == 'None':
                return np.array([True]*self._matches_df.shape[0])
            else:
                odd, cond = filter_str.split(':')
                if cond[0] == '>':
                    return self._matches_df[odd] > float(cond[1:])
                else:
                    cond_min, cond_max = cond[1: -1].split('-')
                    return (self._matches_df[odd] >= float(cond_min)) & (self._matches_df[odd] <= float(cond_max))

        def get_percentile_mask_and_compute_percentile_probs() -> np.ndarray:
            def get_mask_and_prob(correct_ids: bool, target_col: int, percentile: int):
                if percentile == 0:
                    return self._predicted_targets == target_col, 0.0
                elif percentile == 101:
                    return np.array([False]*self._predicted_targets.shape[0]), 1.1
                else:
                    correct_proba = self._predicted_probabilities[correct_ids & (self._targets == target_col), target_col]

                    if correct_proba.shape[0] == 0:
                        return [False]*self._predicted_targets.shape[0], 1.1

                    calib_prob = np.percentile(a=correct_proba, q=percentile)
                    percentile_mask = self._predicted_probabilities[:, target_col] >= calib_prob
                    return percentile_mask, calib_prob

            correct_ids = self._targets == self._predicted_targets
            task = self._task_var.get()

            if task == 'Result':
                home_mask, self._home_percent_prob = get_mask_and_prob(correct_ids=correct_ids, target_col=0, percentile=self._home_percentile_slider.get_value())
                draw_mask, self._draw_percent_prob = get_mask_and_prob(correct_ids=correct_ids, target_col=1, percentile=self._draw_percentile_slider.get_value())
                away_mask, self._away_percent_prob = get_mask_and_prob(correct_ids=correct_ids, target_col=2, percentile=self._away_percentile_slider.get_value())
                return (home_mask | draw_mask) | away_mask
            elif task == 'Over':
                under_mask, self._under_percent_prob = get_mask_and_prob(correct_ids=correct_ids, target_col=0, percentile=self._under_percentile_slider.get_value())
                over_mask, self._over_percent_prob = get_mask_and_prob(correct_ids=correct_ids, target_col=1, percentile=self._over_percentile_slider.get_value())
                return under_mask | over_mask
            else:
                raise NotImplementedError(f'Undefined task: "{task}"')

        def display_metrics(targets: np.ndarray, predictions: np.ndarray):
            if mask.sum() == 0:
                self._acc_var.set(value=0.0)
                self._f1_var.set(value=0.0)
                self._prec_var.set(value=0.0)
                self._rec_var.set(value=0.0)
            else:
                task = self._task_var.get()

                if task == 'Result':
                    average = 'macro'
                else:
                    assert task == 'Over', f'Undefined task: {task}'

                    average = 'binary'

                self._acc_var.set(value=round(accuracy_score(y_true=targets, y_pred=predictions), 2))
                self._f1_var.set(value=round(f1_score(y_true=targets, y_pred=predictions, average=average, zero_division=0.0), 2))
                self._prec_var.set(value=round(precision_score(y_true=targets, y_pred=predictions, average=average, zero_division=0.0), 2))
                self._rec_var.set(value=round(recall_score(y_true=targets, y_pred=predictions, average=average, zero_division=0.0), 2))

        def display_matches(matches_df: pd.DataFrame, targets: np.ndarray, predictions: np.ndarray, predicted_probabilities: np.ndarray):
            if matches_df.shape[0] > 0:
                match_columns = ['Date', 'Home Team', 'Away Team'] + [col for col in ['1', 'X', '2'] if col in self._matches_df.columns]
                selected_matches = matches_df[match_columns]
                selected_matches.insert(loc=0, column='Index', value=np.arange(1, selected_matches.shape[0] + 1))
                selected_matches['Result'] = targets
                selected_matches['Predicted'] = predictions

                task = self._task_var.get()

                if task == 'Result':
                    assert self._predicted_probabilities.shape[1] == 3, 'Incorrect probabilities passed into _add_matches function'

                    selected_matches['Result'] = selected_matches['Result'].replace({0: 'H', 1: 'D', 2: 'A'})
                    selected_matches['Predicted'] = selected_matches['Predicted'].replace({0: 'H', 1: 'D', 2: 'A'})
                    selected_matches['Prob-H'] = predicted_probabilities[:, 0]
                    selected_matches['Prob-D'] = predicted_probabilities[:, 1]
                    selected_matches['Prob-A'] = predicted_probabilities[:, 2]
                elif task == 'Over':
                    assert self._predicted_probabilities.shape[1] == 2, 'Incorrect probabilities passed into _add_matches function'

                    selected_matches['Result'] = selected_matches['Result'].replace({0: 'U(2.5)', 1: 'O(2.5)'})
                    selected_matches['Predicted'] = selected_matches['Predicted'].replace({0: 'U(2.5)', 1: 'O(2.5)'})
                    selected_matches['Prob-U(2.5)'] = predicted_probabilities[:, 0]
                    selected_matches['ProbO(2.5)'] = predicted_probabilities[:, 1]
                else:
                    raise NotImplementedError(f'Not implemented task: {task}')

                for i, values in enumerate(selected_matches.values.tolist()):
                    self._treeview.insert(parent='', index=i, values=values)

                highlight_mask = predictions == targets
                self._highlight_treeview_items(mask=highlight_mask)

        if self._targets is None or not self._model_id_var.get():
            return

        mask = get_odd_mask() & get_percentile_mask_and_compute_percentile_probs()

        assert mask.shape[0] == self._targets.shape[0], f'Mask size does not equal targets size: {mask.shape[0]} vs {self._targets.shape[0]}'

        matches_df = self._matches_df[mask]
        targets = self._targets[mask]
        predictions = self._predicted_targets[mask]
        predicted_probabilities = self._predicted_probabilities[mask]

        clear_treeview_items()
        display_metrics(targets=targets, predictions=predictions)
        display_matches(matches_df=matches_df, targets=targets, predictions=predictions, predicted_probabilities=predicted_probabilities)

    def _highlight_treeview_items(self, mask: np.ndarray):
        selected_items = self._treeview.selection()

        if len(selected_items) > 0:
            self._treeview.selection_remove(selected_items)

        if len(mask) > 0:
            items = self._treeview.get_children()

            assert len(items) == mask.shape[0], f'Mask size does not equal number of matches: {mask.shape[0]} vs {len(items)}'

            selections = [item for item, is_correct in zip(items, mask) if is_correct]

            if selections:
                self._treeview.selection_set(selections)

    def _init_dialog(self):
        messagebox.showinfo(
            parent=self.window,
            title='Filters',
            message='Use odd filter to select rows (matches) that satisfy a filter condition. (e.g. all matches where home prob 1 > 3.10).'
                    '\nUse percentile filters to select matches with minimum predicted probabilities (e.g. all home prob > 5% of all home probs).'
                    '\nYou can disable a percentile filter by setting it to 101 (e.g. highlight no matches)'
                    '\nFor more info, refer to help menu.'
        )

    def _get_dialog_result(self):
        return None
