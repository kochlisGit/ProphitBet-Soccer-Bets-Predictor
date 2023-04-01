import threading
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Callable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tkinter import messagebox, IntVar, StringVar, scrolledtext, Scale, END, INSERT
from tkinter.ttk import Button, Combobox, Entry, Label
from database.repositories.model import ModelRepository
from gui.dialogs.dialog import Dialog
from gui.dialogs.model.utils import display_eval_metrics
from gui.dialogs.task import TaskDialog
from gui.widgets.utils import create_tooltip_btn, validate_float_positive_entry
from models.model import Model
from models.scikit.rf import RandomForest
from models.tf.nn import FCNet
from tuners.tuner import Tuner
from tuners.scikit.rf import RandomForestTuner
from tuners.tf.nn import FCNetTuner


class TuningDialog(Dialog, ABC):
    def __init__(
            self,
            root,
            title: str,
            window_size: dict,
            model_repository: ModelRepository,
            league_name: str,
            random_seed: int,
            matches_df: pd.DataFrame,
            one_hot: bool
    ):
        super().__init__(root=root, title=title, window_size=window_size)

        self._model_repository = model_repository
        self._league_name = league_name
        self._random_seed = random_seed
        self._matches_df = matches_df
        self._one_hot = one_hot

        self._metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
        self._metric_targets = {'Home': 0, 'Draw': 1, 'Away': 2}
        self._best_params = None
        self._eval_metrics = None

        self._n_trials_var = IntVar(value=100)
        self._metric_var = StringVar(value='Accuracy')
        self._metric_target_var = StringVar(value='Home')
        self._num_eval_samples_var = IntVar(value=50)
        self._tune_btn = Button(self.window, text='Tune', command=self._submit_tuning)
        self.window.bind('<Return>', lambda event: self._submit_tuning())

    @property
    def n_trials_var(self) -> IntVar:
        return self._n_trials_var

    @property
    def metrics(self) -> list:
        return self._metrics

    @property
    def metric_targets(self) -> dict:
        return self._metric_targets

    @property
    def metric_var(self) -> StringVar:
        return self._metric_var

    @property
    def metric_target_var(self) -> StringVar:
        return self._metric_target_var

    @property
    def num_eval_samples_var(self) -> IntVar:
        return self._num_eval_samples_var

    @property
    def tune_btn(self) -> Button:
        return self._tune_btn

    def _tune_fn(self):
        task_dialog = TaskDialog(self._window, self._title)
        task_thread = threading.Thread(target=self._tune, args=(task_dialog,))
        task_thread.start()
        task_dialog.open()

    def _tune(self, task_dialog: TaskDialog):
        metric_name = self._metric_var.get()
        metric_target = self._metric_targets[self._metric_target_var.get()]

        if metric_name == 'Accuracy':
            metric = lambda y_true, y_pred: accuracy_score(y_true=y_true, y_pred=y_pred)
        elif metric_name == 'F1':
            metric = lambda y_true, y_pred: f1_score(y_true=y_true, y_pred=y_pred, average=None)[metric_target]
        elif metric_name == 'Precision':
            metric = lambda y_true, y_pred: precision_score(
                y_true=y_true, y_pred=y_pred, average=None)[metric_target]
        elif metric_name == 'Recall':
            metric = lambda y_true, y_pred: recall_score(
                y_true=y_true, y_pred=y_pred, average=None)[metric_target]
        else:
            task_dialog.close()
            raise NotImplementedError(f'Error: Metric "{metric_name}" has not been implemented yet')

        tuner = self._construct_tuner(
            n_trials=self._n_trials_var.get(),
            metric=metric,
            matches_df=self._matches_df,
            num_eval_samples=self._num_eval_samples_var.get(),
            random_seed=self._random_seed
        )
        self._best_params = tuner.tune()
        self._train(
            x_train=tuner.x_train,
            y_train=tuner.y_train,
            x_test=tuner.x_test,
            y_test=tuner.y_test,
            random_seed=self._random_seed,
            best_params=self._best_params
        )
        task_dialog.close()

    def _train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray,
            random_seed: int,
            best_params: dict
    ):
        model = self._construct_model(input_shape=x_train.shape[1:], random_seed=random_seed)
        self._build_model(model=model, best_params=best_params)
        self._eval_metrics = model.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            use_over_sampling=best_params['user_over_sampling']
        )
        self._model_repository.store_model(model=model, league_name=self._league_name)

    def _submit_tuning(self):
        if messagebox.askokcancel(
                'Training Confirmation',
                'Training has been submitted. You will not be able to train '
                'another model until this session finishes. Do you wish to continue?'
        ):
            self._tune_btn['state'] = 'disabled'
            self._tune_fn()
            self._tune_btn['state'] = 'enabled'

            if self._eval_metrics is not None and self._best_params is not None:
                self._display_best_params(best_params=self._best_params)
                display_eval_metrics(self._eval_metrics)
                self._eval_metrics = None
                self._best_params = None

    @abstractmethod
    def _construct_tuner(
            self,
            n_trials: int,
            metric: Callable,
            matches_df: pd.DataFrame,
            num_eval_samples: int,
            random_seed: int = 0
    ) -> Tuner:
        pass

    @abstractmethod
    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        pass

    @abstractmethod
    def _build_model(self, model: Model, best_params: dict):
        pass

    @abstractmethod
    def _display_best_params(self, best_params: dict):
        pass

    def _dialog_result(self) -> None:
        return None


class TuningNNDialog(TuningDialog):
    def __init__(
            self,
            root,
            model_repository: ModelRepository,
            league_name: str,
            random_seed: int,
            matches_df: pd.DataFrame
    ):
        super().__init__(
            root=root,
            title='Neural Network Tuning',
            window_size={'width': 1020, 'height': 800},
            model_repository=model_repository,
            league_name=league_name,
            random_seed=random_seed,
            matches_df=matches_df,
            one_hot=True
        )

        self._epochs_var = IntVar(value=80)
        self._early_stopping_epochs_var = IntVar(value=35)
        self._learning_rate_decay_factor_var = StringVar(value='0.2')
        self._learning_rate_decay_epochs_var = IntVar(value=10)
        self._min_layers_var = IntVar(value=3)
        self._max_layers_var = IntVar(value=5)
        self._min_units_var = IntVar(value=32)
        self._max_units_var = IntVar(value=128)
        self._units_increment_var = IntVar(value=16)

        self._text = None

    def _initialize(self):
        validate_float = self.window.register(validate_float_positive_entry)

        Label(self.window, text='Trials', font=('Arial', 10)).place(x=20, y=15)
        Label(self.window, text='Metric', font=('Arial', 10)).place(x=20, y=70)
        Label(self.window, text='Target to Maximize', font=('Arial', 10)).place(x=20, y=110)
        Label(self.window, text='Evaluation Samples', font=('Arial', 10)).place(x=20, y=165)
        Label(self.window, text='Epochs', font=('Arial', 10)).place(x=20, y=220)
        Label(self.window, text='Early Stopping Epochs', font=('Arial', 10)).place(x=20, y=275)
        Label(self.window, text='Learning Rate Decay Factor', font=('Arial', 10)).place(x=20, y=325)
        Label(self.window, text='Learning Rate Decay Epochs', font=('Arial', 10)).place(x=20, y=385)
        Label(self.window, text='Min Layers', font=('Arial', 10)).place(x=580, y=165)
        Label(self.window, text='Max Layers', font=('Arial', 10)).place(x=580, y=220)
        Label(self.window, text='Min Neurons', font=('Arial', 10)).place(x=580, y=275)
        Label(self.window, text='Max Neurons', font=('Arial', 10)).place(x=580, y=325)
        Label(self.window, text='Neuron Increment', font=('Arial', 10)).place(x=580, y=385)

        create_tooltip_btn(
            root=self.window, x=240, y=15,
            text='Number of search trials (Iterations). Should be positive integer'
        )
        create_tooltip_btn(
            root=self.window, x=240, y=70,
            text='The metric that the tuning algorithm tries to maximize.'
        )
        create_tooltip_btn(
            root=self.window, x=240, y=110,
            text='The metric target (Home/Draw/Away) of the metric.'
                 '\nFor example, if Home, then the algorithm tries to maximize the metric of '
                 '\n"Home" results. If metric is "Accuracy", this option is ignored and'
                 '\nthe average accuracy for all targets is maximized'
        )
        create_tooltip_btn(
            root=self.window, x=240, y=165,
            text='Number of evaluation samples to exclude from training'
                 '\nand use them as evaluation samples'
        )
        create_tooltip_btn(
            root=self.window, x=240, y=220,
            text='Number of training epochs. Should be positive integer, usually (50-150)'
        )
        create_tooltip_btn(
            root=self.window, x=240, y=275,
            text='Number early stopping epochs to wait before stopping training,'
                 '\nif validation loss does not improve. Should be positive integer or zero,'
                 '\n usually (25-50). Set it to 0 to disable Early Stopping mechanism'
        )
        create_tooltip_btn(
            root=self.window, x=240, y=325,
            text='Learning rate decay factor to reduce learning rate,'
                 '\nif validation loss does not improve. Should be positive float or zero between '
                 '\n0.0 and 1.0, usually (0.1-0.4). Set it to 0 to disable Learning Rate Decay'
        )
        create_tooltip_btn(
            root=self.window, x=240, y=385,
            text='Number of epochs to wait for validation loss improvement,'
                 '\nbefore reducing learning rate. Should be positive integer or zero, usually (5-15).'
                 '\nSet it to 0 to disable Learning Rate Decay'
        )
        create_tooltip_btn(
            root=self.window, x=720, y=165,
            text='Minimum number of hidden layers'
        )
        create_tooltip_btn(
            root=self.window, x=720, y=220,
            text='Maximum number of hidden layers'
        )
        create_tooltip_btn(
            root=self.window, x=720, y=275,
            text='Minimum number of units (Neurons)'
        )
        create_tooltip_btn(
            root=self.window, x=720, y=325,
            text='Maximum number of units (Neurons)'
        )
        create_tooltip_btn(
            root=self.window, x=720, y=385,
            text='Increment of units during each trial'
        )

        Scale(
            self.window, from_=1, to=2000, tickinterval=100,
            orient='horizontal', length=500, variable=self.n_trials_var
        ).place(x=315, y=1)

        metric_cb = Combobox(
            self.window, width=10, font=('Arial', 10), state='readonly', textvariable=self.metric_var
        )
        metric_cb['values'] = self.metrics
        metric_cb.current(0)
        metric_cb.place(x=315, y=70)

        metric_target_cb = Combobox(
            self.window, width=10, font=('Arial', 10), state='readonly', textvariable=self.metric_target_var
        )
        metric_target_cb['values'] = list(self.metric_targets.keys())
        metric_target_cb.current(0)
        metric_target_cb.place(x=315, y=110)

        Scale(
            self.window, from_=0, to=250, tickinterval=50, orient='horizontal',
            length=280, variable=self.num_eval_samples_var
        ).place(x=315, y=145)

        Scale(
            self.window, from_=1, to=251, tickinterval=50, orient='horizontal', length=220, variable=self._epochs_var
        ).place(x=315, y=205)

        Scale(
            self.window, from_=0, to=50, tickinterval=10, orient='horizontal',
            length=140, variable=self._early_stopping_epochs_var
        ).place(x=315, y=260)

        Entry(
            self.window, width=10, font=('Arial', 10),
            validate='key', validatecommand=(validate_float, '%P'), textvariable=self._learning_rate_decay_factor_var
        ).place(x=315, y=325)

        Scale(
            self.window, from_=0, to=30, tickinterval=10, orient='horizontal',
            length=140, variable=self._learning_rate_decay_epochs_var
        ).place(x=315, y=360)

        Scale(
            self.window, from_=1, to=5, tickinterval=1, orient='horizontal',
            length=150, variable=self._min_layers_var
        ).place(x=775, y=145)

        Scale(
            self.window, from_=1, to=5, tickinterval=1, orient='horizontal', length=150, variable=self._max_layers_var
        ).place(x=775, y=205)

        Scale(
            self.window, from_=8, to=32, tickinterval=8, orient='horizontal',
            length=200, variable=self._min_units_var
        ).place(x=775, y=260)

        Scale(
            self.window, from_=32, to=532, tickinterval=100, orient='horizontal',
            length=200, variable=self._max_units_var
        ).place(x=775, y=315)

        Scale(
            self.window, from_=1, to=32, tickinterval=10, orient='horizontal',
            length=140, variable=self._units_increment_var
        ).place(x=775, y=370)

        self.tune_btn.place(x=490, y=425)

        self._text = scrolledtext.ScrolledText(self.window, width=60, height=19, state='disabled')
        self._text.place(x=285, y=470)

    def _construct_tuner(
            self,
            n_trials: int,
            metric: Callable,
            matches_df: pd.DataFrame,
            num_eval_samples: int,
            random_seed: int = 0
    ) -> Tuner:
        return FCNetTuner(
            n_trials=n_trials,
            metric=metric,
            matches_df=matches_df,
            num_eval_samples=num_eval_samples,
            epochs=self._epochs_var.get(),
            early_stopping_epochs=self._early_stopping_epochs_var.get(),
            learning_rate_decay_factor=float(self._learning_rate_decay_factor_var.get()),
            learning_rate_decay_epochs=self._learning_rate_decay_epochs_var.get(),
            min_layers=self._min_layers_var.get(),
            max_layers=self._max_layers_var.get(),
            min_units=self._min_units_var.get(),
            max_units=self._max_units_var.get(),
            units_increment=self._units_increment_var.get(),
            random_seed=random_seed
        )

    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        return FCNet(input_shape=input_shape, random_seed=random_seed)

    def _build_model(self, model: Model, best_params: dict):
        num_hidden_layers = best_params['num_hidden_layers']
        hidden_layers = [best_params[f'layer_{i}'] for i in range(num_hidden_layers)]
        activations = [best_params[f'activation_{i}'] for i in range(num_hidden_layers)]
        batch_normalizations = [best_params[f'bn_{i}'] for i in range(num_hidden_layers)]
        regularizations = [best_params[f'regularization_{i}'] for i in range(num_hidden_layers)]
        dropouts = [best_params[f'dropout_{i}'] for i in range(num_hidden_layers)]

        model.build_model(
            epochs=self._epochs_var.get(),
            batch_size=best_params['batch_size'],
            early_stopping_epochs=self._early_stopping_epochs_var.get(),
            learning_rate_decay_factor=float(self._learning_rate_decay_factor_var.get()),
            learning_rate_decay_epochs=self._learning_rate_decay_epochs_var.get(),
            learning_rate=best_params['learning_rate'],
            noise_range=best_params['noise_range'],
            hidden_layers=hidden_layers,
            batch_normalizations=batch_normalizations,
            activations=activations,
            regularizations=regularizations,
            dropouts=dropouts,
            optimizer=best_params['optimizer']
        )

    def _display_best_params(self, best_params: dict):
        s = '{'
        for param_name, param_value in best_params.items():
            s += f'\n\tParameter: {param_name} = {param_value}'
        s += '\n}'

        self._text['state'] = 'normal'
        self._text.delete(1.0, END)
        self._text.insert(INSERT, s)
        self._text['state'] = 'disabled'


class TuningRFDialog(TuningDialog):
    def __init__(
            self,
            root,
            model_repository: ModelRepository,
            league_name: str,
            random_seed: int,
            matches_df: pd.DataFrame
    ):
        super().__init__(
            root=root,
            title='Random Forest Tuning',
            window_size={'width': 520, 'height': 600},
            model_repository=model_repository,
            league_name=league_name,
            random_seed=random_seed,
            matches_df=matches_df,
            one_hot=False
        )

        self._text = None

    def _initialize(self):
        Label(self.window, text='Trials', font=('Arial', 10)).place(x=20, y=15)
        Label(self.window, text='Metric', font=('Arial', 10)).place(x=20, y=65)
        Label(self.window, text='Target to Maximize', font=('Arial', 10)).place(x=20, y=115)
        Label(self.window, text='Evaluation Samples', font=('Arial', 10)).place(x=20, y=165)

        create_tooltip_btn(
            root=self.window, x=220, y=15,
            text='Number of search trials (Iterations). Should be positive integer'
        )
        create_tooltip_btn(
            root=self.window, x=220, y=65,
            text='The metric that the tuning algorithm tries to maximize.'
        )
        create_tooltip_btn(
            root=self.window, x=220, y=115,
            text='The metric target (Home/Draw/Away) of the metric.'
                 '\nFor example, if Home, then the algorithm tries to maximize the metric of '
                 '\n"Home" results. If metric is "Accuracy", this option is ignored and'
                 '\nthe average accuracy for all targets is maximized'
        )
        create_tooltip_btn(
            root=self.window, x=220, y=165,
            text='Number of evaluation samples to exclude from training'
                 '\nand use them as evaluation samples'
        )

        Scale(
            self.window, from_=1, to=500, tickinterval=100,
            orient='horizontal', length=220, variable=self.n_trials_var
        ).place(x=280, y=1)

        metric_cb = Combobox(
            self.window, width=10, font=('Arial', 10), state='readonly', textvariable=self.metric_var
        )
        metric_cb['values'] = self.metrics
        metric_cb.current(0)
        metric_cb.place(x=280, y=65)

        metric_target_cb = Combobox(
            self.window, width=10, font=('Arial', 10), state='readonly', textvariable=self.metric_target_var
        )
        metric_target_cb['values'] = list(self.metric_targets.keys())
        metric_target_cb.current(0)
        metric_target_cb.place(x=280, y=115)

        Scale(
            self.window, from_=0, to=250, tickinterval=50, orient='horizontal',
            length=220, variable=self.num_eval_samples_var
        ).place(x=280, y=150)

        self.tune_btn.place(x=200, y=235)

        self._text = scrolledtext.ScrolledText(self.window, width=55, height=18, state='disabled')
        self._text.place(x=35, y=280)

    def _construct_tuner(
            self,
            n_trials: int,
            metric: Callable,
            matches_df: pd.DataFrame,
            num_eval_samples: int,
            random_seed: int = 0
    ) -> Tuner:
        return RandomForestTuner(
            n_trials=n_trials,
            metric=metric,
            matches_df=matches_df,
            num_eval_samples=num_eval_samples,
            random_seed=random_seed
        )

    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        return RandomForest(input_shape=input_shape, random_seed=random_seed)

    def _build_model(self, model: Model, best_params: dict):
        model.build_model(
            n_estimators=best_params['n_estimators'],
            max_features=best_params['max_features'],
            max_depth=best_params['max_depth'],
            min_samples_leaf=best_params['min_samples_leaf'],
            min_samples_split=best_params['min_samples_split'],
            bootstrap=best_params['bootstrap'],
            class_weight=best_params['class_weight'],
            is_calibrated=best_params['is_calibrated']
        )

    def _display_best_params(self, best_params: dict):
        s = '{'
        for param_name, param_value in best_params.items():
            s += f'\n\tParameter: {param_name} = {param_value}'
        s += '\n}'

        self._text['state'] = 'normal'
        self._text.delete(1.0, END)
        self._text.insert(INSERT, s)
        self._text['state'] = 'disabled'
