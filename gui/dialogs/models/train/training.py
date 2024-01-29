import pandas as pd
import config
from abc import ABC, abstractmethod
from tabulate import tabulate
from tkinter import BooleanVar, StringVar, IntVar, DoubleVar, scrolledtext, Scale, messagebox, END
from tkinter.ttk import Button, Checkbutton, Combobox, Entry, Label, Separator
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.dialog import Dialog
from gui.dialogs.analysis.tuning import TuningImportancePlotter
from gui.task import TaskDialog
from gui.widgets.tunable import TunableWidget
from gui.widgets.utils import create_tooltip_btn, validate_id_entry
from models.tasks import ClassificationTask
from models.trainer import Trainer
from models.tuner import Tuner


class TrainingDialog(Dialog, ABC):
    def __init__(
            self,
            root,
            title: str,
            matches_df: pd.DataFrame,
            league_config: LeagueConfig,
            model_repository: ModelRepository
    ):
        super().__init__(root=root, title=title, window_size={'width': 1000, 'height': 750})

        self._matches_df = matches_df.dropna().reset_index(drop=True)
        self._league_config = league_config
        self._model_repository = model_repository

        self._trainer = Trainer(model_repository=model_repository, fit_test_size=config.fit_test_size)
        self._tuner = Tuner(trainer=self._trainer)
        self._tasks = {
            'Result': ClassificationTask.Result,
            'U/O (2.5)': ClassificationTask.Over
        }
        self._metrics = ['accuracy', 'f1', 'precision', 'recall']
        self._training_running = False

        self._task_var = StringVar(value='Result')
        self._model_id_var = StringVar()
        self._n_trials_var = IntVar(value=500)
        self._metric_var = StringVar(value='accuracy')

        self._textarea = None

        self._tunable_widgets = {}

    @abstractmethod
    def _get_model_cls(self):
        pass

    def _add_tunable_widget(
            self,
            key: str,
            widget_cls,
            param_values: dict or tuple or list,
            value_variable: StringVar or BooleanVar or IntVar or DoubleVar,
            name: str,
            description: str or None,
            x: int,
            y: int,
            x_pad: int,
            **widget_params
    ):
        self._tunable_widgets[key] = TunableWidget(
            widget_cls=widget_cls,
            param_values=param_values,
            value_variable=value_variable,
            window=self.window,
            name=name,
            description=description,
            x=x,
            y=y,
            x_pad=x_pad,
            **widget_params
        )

    def _set_best_params(self, best_params: dict):
        for param_name, value in best_params.items():
            self._tunable_widgets[param_name].enable()
            self._tunable_widgets[param_name].set_value(value=value)

    def _init_dialog(self):
        Label(
            self.window, text='Task:', font=('Arial', 12)
        ).place(x=260, y=20)
        Combobox(
            self.window, values=list(self._tasks.keys()), state='readonly', font=('Arial', 10), textvariable=self._task_var, width=10
        ).place(x=320, y=20)

        Label(
            self.window, text='Model ID:', font=('Arial', 12)
        ).place(x=450, y=20)
        Entry(
            self.window,
            width=17,
            font=('Arial', 10),
            textvariable=self._model_id_var
        ).place(x=530, y=20)
        create_tooltip_btn(
            root=self.window,
            text='Unique Model ID that will be stored in database'
        ).place(x=670, y=20)

        Label(
            self.window, text='Tune Trials:', font=('Arial', 12)
        ).place(x=180, y=75)
        Scale(
            self.window, from_=1, to=1000, tickinterval=200, orient='horizontal', length=180, variable=self._n_trials_var
        ).place(x=280, y=55)
        create_tooltip_btn(
            root=self.window,
            text='Number of search trials (iterations)'
        ).place(x=475, y=70)

        Label(
            self.window, text='Tune Metric:', font=('Arial', 12)
        ).place(x=525, y=70)
        Combobox(
            self.window, values=self._metrics, state='readonly', font=('Arial', 10), textvariable=self._metric_var, width=10
        ).place(x=630, y=70)
        create_tooltip_btn(
            root=self.window,
            text='Optimization metric for tuner (Tuner will try to maximize it)'
        ).place(x=740, y=70)

        Separator(self.window, orient='horizontal').place(x=0, y=120, relwidth=1)

        normalizer_var = StringVar(value='Standard')
        widget_params = {
            'master': self.window, 'values': config.normalizers, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': normalizer_var, 'width': 10
        }
        self._add_tunable_widget(
            key='normalizer',
            widget_cls=Combobox,
            param_values=config.normalizers,
            value_variable=normalizer_var,
            name='Normalizer',
            description='Input normalization method',
            x=20,
            y=140,
            x_pad=15,
            **widget_params
        )

        sampler_var = StringVar(value='None')
        widget_params = {
            'master': self.window, 'values': config.samplers, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': sampler_var, 'width': 10
        }
        self._add_tunable_widget(
            key='sampler',
            widget_cls=Combobox,
            param_values=config.samplers,
            value_variable=sampler_var,
            name='Sampler',
            description='Input re-sampling method for dealing with imbalanced targets',
            x=340,
            y=140,
            x_pad=15,
            **widget_params
        )

        calibration_var = BooleanVar(value=True)
        widget_params = {
            'master': self.window, 'text': '', 'offvalue': False, 'onvalue': True, 'variable': calibration_var
        }
        self._add_tunable_widget(
            key='calibrate_probabilities',
            widget_cls=Checkbutton,
            param_values=[True, False],
            value_variable=calibration_var,
            name='Calibrate Probabilities',
            description='Calibrate output probabilities of model',
            x=660,
            y=140,
            x_pad=15,
            **widget_params
        )

        Separator(self.window, orient='horizontal').place(x=0, y=490, relwidth=1)

        Button(self.window, text='Train', command=self._train).place(x=450, y=510)
        self._textarea = scrolledtext.ScrolledText(self.window, width=120, height=12, state='disabled')
        self._textarea.place(x=15, y=550)

    def _train(self):
        if self._training_running:
            messagebox.showerror(parent=self.window, title='Training on Progress', message='Wait until previous training process is completed')
            return

        self._training_running = True

        model_id = self._model_id_var.get()
        task = self._task_var.get()
        model_configs = self._model_repository.get_model_configs(league_id=self._league_config.league_id)

        if not validate_id_entry(parent=self.window, text=model_id):
            self._training_running = False
            return
        if task in model_configs and model_id in model_configs[task]:
            overwrite_result = messagebox.askyesno(
                parent=self.window,
                title='Model Exists',
                message=f'Model "{model_id}" already exists. Do you want to overwrite it?'
            )
            if not overwrite_result:
                self._training_running = False
                return

        model_params = {}
        tune_params = {}
        for key_param, widget in self._tunable_widgets.items():
            if widget.is_tunable():
                tune_params[key_param] = widget.param_values
            else:
                model_params[key_param] = widget.get_value()

        if len(tune_params) == 0:
            results = self._train_model(model_params=model_params)
            self._show_results(results=results)
        else:
            best_params, proceed_result = self._tune_model(model_params=model_params, tune_params=tune_params)
            self._set_best_params(best_params=best_params)

            if proceed_result:
                model_params.update(best_params)
                results = self._train_model(model_params=model_params)
                self._show_results(results=results)

        self._training_running = False

    def _train_model(self, model_params: dict) -> str:
        model_id = self._model_id_var.get()
        league_id = self._league_config.league_id
        task = self._tasks[self._task_var.get()]

        evaluation_dict = TaskDialog(
            master=self.window,
            title='Cross Validation',
            task=self._trainer.cross_validate,
            args=(self._matches_df, league_id, model_id, task, self._get_model_cls(), model_params)
        ).start()

        results = f'--- Cross Validation ---\n{tabulate([list(evaluation_dict.keys()), list(evaluation_dict.values())])}'
        proceed_result = messagebox.askyesno(
            parent=self.window,
            title='Cross Validation Evaluation',
            message=f'{results}\nDo you wish to continue training?'
        )

        if not proceed_result:
            return results

        save_model = True
        _, _, evaluation_dict, classification_report = TaskDialog(
            master=self.window,
            title='Training Model',
            task=self._trainer.fit,
            args=(self._matches_df, league_id, model_id, task, save_model, self._get_model_cls(), model_params)
        ).start()

        results = '--- Evaluation ---' \
                  f'\n{tabulate([list(evaluation_dict.keys()), list(evaluation_dict.values())])} ' \
                  '\n\n--- Classification Report --- ' \
                  f'\n{classification_report}\nModel: {model_id} has been created.'
        messagebox.showinfo(
            parent=self.window,
            title='Fit Results',
            message=results
        )
        return results

    def _tune_model(self, tune_params: dict, model_params: dict) -> (dict, bool):
        metric = self._metric_var.get()
        n_trials = self._n_trials_var.get()
        model_id = self._model_id_var.get()
        task = self._tasks[self._task_var.get()]

        tune_args = (
            n_trials,
            metric,
            self._matches_df,
            self._league_config.league_id,
            model_id,
            task,
            self._get_model_cls(),
            model_params,
            tune_params
        )
        study = TaskDialog(
            master=self.window,
            title='Tuning Model',
            task=self._tuner.tune,
            args=tune_args
        ).start()

        proceed_result = messagebox.askyesno(
            parent=self.window,
            title=f'Tuning Results',
            message=f'Best tuning score ("{metric}") found: {study.best_value}. Do you wish to continue training?'
        )

        if n_trials > 1:
            plot_importance_result = messagebox.askyesno(
                parent=self.window,
                title=f'Tuned Param Importance',
                message=f'Do you wish to plot the importance scores of the tuned parameters?'
            )
        else:
            plot_importance_result = False

        if plot_importance_result:
            try:
                tune_param_importance_scores = self._tuner.get_param_importance_scores(study=study)
                TuningImportancePlotter(root=self.window, importance_scores=tune_param_importance_scores, task=task).open_and_wait()
            except Exception as e:
                messagebox.showerror(parent=self.window, title='Failed to generate plot', message=str(e))
        return study.best_params, proceed_result

    def _show_results(self, results: str):
        self._textarea.config(state='normal')

        self._textarea.delete('1.0', END)
        self._textarea.insert(END, results)

        self._textarea.config(state='disabled')
