import pandas as pd
from tkinter import StringVar, IntVar, DoubleVar, BooleanVar, Scale, messagebox
from tkinter.ttk import Combobox, Entry, Checkbutton
from database.entities.leagues.league import LeagueConfig
from database.repositories.model import ModelRepository
from gui.dialogs.models.train.training import TrainingDialog
from gui.widgets.intslider import IntSlider
from models.estimators import NeuralNetwork


class NeuralNetworkTrainDialog(TrainingDialog):
    def __init__(self, root, matches_df: pd.DataFrame, league_config: LeagueConfig, model_repository: ModelRepository):
        super().__init__(
            root=root,
            title='Neural Network Trainer',
            matches_df=matches_df,
            league_config=league_config,
            model_repository=model_repository
        )

        self._min_hidden_layers = 1
        self._max_hidden_layers = 5
        self._fc_hidden_units_range = (32, 512, 32)
        self._activations_list = ['sigmoid', 'tanh', 'relu', 'elu', 'gelu']
        self._weight_regularizations_list = ['None', 'l1', 'l2', 'l1_l2']
        self._batch_normalization_list = [False, True]
        self._dropout_rate_range = (0.0, 0.8, 0.1)
        self._batch_size_range = (8, 64, 8)
        self._epochs_range = (20, 200, 10)
        self._early_stopping_patience_range = (0, 50, 10)
        self._learning_rate_range = (0.0005, 0.015, 0.001)
        self._learning_rate_patience_range = (0, 20, 5)
        self._label_smoothing_list = [False, True]
        self._optimizers_list = ['adam', 'radam', 'adabelief', 'lookahead-adabelief']
        self._input_noise_range = (0.0, 0.25, 0.05)

    def _get_model_cls(self):
        return NeuralNetwork

    def _init_dialog(self):
        self._model_id_var.set(value='neural-network-model')

        super()._init_dialog()

        self._tunable_widgets['calibrate_probabilities'].set_value(value=False)

    def _create_widgets(self):
        fc_hiddens_var = StringVar(value='64,128,128,32')
        widget_params = {
            'master': self.window, 'width': 18, 'font': ('Arial', 10), 'textvariable': fc_hiddens_var
        }
        self._add_tunable_widget(
            key='fc_hiddens',
            widget_cls=Entry,
            param_values=self._fc_hidden_units_range,
            value_variable=fc_hiddens_var,
            name='FC-Hiddens',
            description='Hidden units per layer. Max 5 layers. Layer units are separated by "," with (min 32 units, max 512)',
            x=10,
            y=190,
            x_pad=15,
            **widget_params
        )

        activation_var = StringVar(value='relu')
        widget_params = {
            'master': self.window, 'values': self._activations_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': activation_var, 'width': 10
        }
        self._add_tunable_widget(
            key='activation_fn',
            widget_cls=Combobox,
            param_values=self._activations_list,
            value_variable=activation_var,
            name='Activation',
            description='Activation function for hidden layers. "relu" is usually fast and efficient',
            x=375,
            y=190,
            x_pad=15,
            **widget_params
        )

        weight_reg_var = StringVar(value='None')
        widget_params = {
            'master': self.window, 'values': self._weight_regularizations_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': weight_reg_var, 'width': 10
        }
        self._add_tunable_widget(
            key='weight_regularization',
            widget_cls=Combobox,
            param_values=self._weight_regularizations_list,
            value_variable=weight_reg_var,
            name='Weight Regular...',
            description='Weight regularization for input layers (similar to Logistic Regression & SVM)',
            x=680,
            y=190,
            x_pad=15,
            **widget_params
        )

        batch_norm_var = BooleanVar(value=True)
        widget_params = {
            'master': self.window, 'text': '', 'offvalue': False, 'onvalue': True, 'variable': batch_norm_var
        }
        self._add_tunable_widget(
            key='batch_normalization',
            widget_cls=Checkbutton,
            param_values=self._batch_normalization_list,
            value_variable=batch_norm_var,
            name='Batch Normalization',
            description='Batch normalization after each layer',
            x=10,
            y=250,
            x_pad=15,
            **widget_params
        )

        dropout_var = DoubleVar(value=0.0)
        widget_params = {
            'master': self.window, 'from_': self._dropout_rate_range[0], 'to': self._dropout_rate_range[1], 'tickinterval': 0.2, 'resolution': 0.1, 'orient': 'horizontal', 'length': 150, 'variable': dropout_var
        }
        self._add_tunable_widget(
            key='dropout_rate',
            widget_cls=Scale,
            param_values=self._dropout_rate_range,
            value_variable=dropout_var,
            name='Dropout rate',
            description='Dropout rate after each layer',
            x=307,
            y=250,
            x_pad=15,
            **widget_params
        )

        batch_size_var = IntVar(value=16)
        widget_params = {
            'master': self.window, 'from_': self._batch_size_range[0], 'to': self._batch_size_range[1], 'variable': batch_size_var
        }
        self._add_tunable_widget(
            key='batch_size',
            widget_cls=IntSlider,
            param_values=self._batch_size_range,
            value_variable=batch_size_var,
            name='Batch Size',
            description='Batch size per feed-forward operation',
            x=675,
            y=250,
            x_pad=15,
            **widget_params
        )

        epochs_var = IntVar(value=16)
        widget_params = {
            'master': self.window, 'from_': self._epochs_range[0], 'to': self._epochs_range[1], 'variable': epochs_var
        }
        self._add_tunable_widget(
            key='epochs',
            widget_cls=IntSlider,
            param_values=self._epochs_range,
            value_variable=epochs_var,
            name='Epochs',
            description='Neural Network training iterations (data will be fed "epoch" times to neural network)',
            x=10,
            y=320,
            x_pad=15,
            **widget_params
        )

        early_stopping_patience_var = IntVar(value=0)
        widget_params = {
            'master': self.window, 'from_': self._early_stopping_patience_range[0], 'to': self._early_stopping_patience_range[1], 'variable': early_stopping_patience_var
        }
        self._add_tunable_widget(
            key='early_stopping_patience',
            widget_cls=IntSlider,
            param_values=self._early_stopping_patience_range,
            value_variable=early_stopping_patience_var,
            name='Early Stop',
            description='Stops training if the loss does not improve after K epochs',
            x=313,
            y=320,
            x_pad=15,
            **widget_params
        )

        lr_var = DoubleVar(value=0.0)
        widget_params = {
            'master': self.window, 'from_': self._learning_rate_range[0], 'to': self._learning_rate_range[1], 'tickinterval': 0.005, 'resolution': 0.001, 'orient': 'horizontal', 'length': 150, 'variable': lr_var
        }
        self._add_tunable_widget(
            key='learning_rate',
            widget_cls=Scale,
            param_values=self._learning_rate_range,
            value_variable=lr_var,
            name='Learn rate',
            description='Learning rate of neural networks (similar to XG-Boost)',
            x=635,
            y=320,
            x_pad=15,
            **widget_params
        )

        learning_rate_patience_var = IntVar(value=0)
        widget_params = {
            'master': self.window, 'from_': self._learning_rate_patience_range[0], 'to': self._learning_rate_patience_range[1], 'variable': learning_rate_patience_var
        }
        self._add_tunable_widget(
            key='learning_rate_patience',
            widget_cls=IntSlider,
            param_values=self._learning_rate_patience_range,
            value_variable=learning_rate_patience_var,
            name='Learn Patience',
            description='Reduces learning rate after "K" epochs, if neural network does not improve',
            x=10,
            y=390,
            x_pad=15,
            **widget_params
        )

        input_noise_var = DoubleVar(value=0.0)
        widget_params = {
            'master': self.window, 'from_': self._input_noise_range[0], 'to': self._input_noise_range[1], 'tickinterval': 0.1, 'resolution': 0.05, 'orient': 'horizontal', 'length': 150, 'variable': input_noise_var
        }
        self._add_tunable_widget(
            key='input_noise',
            widget_cls=Scale,
            param_values=self._input_noise_range,
            value_variable=input_noise_var,
            name='Input Noise',
            description='Add random input noise as regularization effect',
            x=350,
            y=390,
            x_pad=15,
            **widget_params
        )

        label_smoothing_var = BooleanVar(value=False)
        widget_params = {
            'master': self.window, 'text': '', 'offvalue': False, 'onvalue': True, 'variable': label_smoothing_var
        }
        self._add_tunable_widget(
            key='label_smoothing',
            widget_cls=Checkbutton,
            param_values=self._label_smoothing_list,
            value_variable=label_smoothing_var,
            name='Label Smoothing',
            description='Add random noise to target probabilities',
            x=710,
            y=390,
            x_pad=15,
            **widget_params
        )

        optimizer_var = StringVar(value='adam')
        widget_params = {
            'master': self.window, 'values': self._optimizers_list, 'state': 'readonly', 'font': ('Arial', 10), 'textvariable': optimizer_var, 'width': 13
        }
        self._add_tunable_widget(
            key='optimizer',
            widget_cls=Combobox,
            param_values=self._optimizers_list,
            value_variable=optimizer_var,
            name='Optimizer',
            description='Neural network training optimize ("adam" is usually fast and efficient)',
            x=320,
            y=450,
            x_pad=15,
            **widget_params
        )

    def _get_fc_hiddens(self, fc_hiddens_str) -> list[int] or None:
        if fc_hiddens_str != '':
            try:
                fc_hiddens_list = fc_hiddens_str.strip().split(',')
                fc_hidden_units = [int(units) for units in fc_hiddens_list]

                num_hiddens = len(fc_hidden_units)

                if num_hiddens > self._max_hidden_layers:
                    fc_hidden_units = fc_hidden_units[: self._max_hidden_layers]
                    messagebox.showwarning(
                        parent=self.window,
                        title='Incorrect FC-Hiddens',
                        message=f'Maximum of 5 layers is allowed. Keeping the first 5 layers.'
                    )

                for i in range(len(fc_hidden_units)):
                    units = fc_hidden_units[i]

                    if units < self._fc_hidden_units_range[0]:
                        fc_hidden_units[i] = self._fc_hidden_units_range[0]
                        messagebox.showwarning(
                            parent=self.window,
                            title='Incorrect FC-Hiddens',
                            message=f'Units of Layer {i + 1} are less than {self._fc_hidden_units_range[0]}. Setting units to {self._fc_hidden_units_range[0]}'
                        )
                    if units > self._fc_hidden_units_range[1]:
                        fc_hidden_units[i] = self._fc_hidden_units_range[1]
                        messagebox.showwarning(
                            parent=self.window,
                            title='Incorrect FC-Hiddens',
                            message=f'Units of Layer {i + 1} are greater than {self._fc_hidden_units_range[1]}. Setting units to {self._fc_hidden_units_range[1]}'
                        )

                return fc_hidden_units
            except Exception as e:
                messagebox.showerror(
                    parent=self.window,
                    title='Incorrect FC-Hiddens',
                    message=
                    f'fc-hiddens units should be integers from {self._min_hidden_layers} to {self._max_hidden_layers}, '
                    f'separated by comma, with NO SPACES in between, got {fc_hiddens_str}. '
                    f'Compiler error: {e}'
                )
        else:
            return None

    def _tune_model(self, tune_params: dict, model_params: dict) -> (dict, bool):
        model_params['verbose'] = False
        model_params['summary'] = False

        if 'fc_hiddens' in model_params:
            model_params['fc_hiddens'] = self._get_fc_hiddens(fc_hiddens_str=model_params['fc_hiddens'])

        best_params, proceed_result = super()._tune_model(tune_params=tune_params, model_params=model_params)

        num_layers = len([param for param in best_params if 'layer_' in param])
        best_params['fc_hiddens'] = [best_params[f'layer_{i}'] for i in range(1, num_layers + 1)]
        return best_params, proceed_result

    def _train_model(self, model_params: dict) -> str:
        fc_hiddens = model_params['fc_hiddens']

        if isinstance(fc_hiddens, str):
            model_params['fc_hiddens'] = self._get_fc_hiddens(fc_hiddens_str=fc_hiddens)

        return super()._train_model(model_params=model_params)

    def _train(self):
        calibrate_prob_widget = self._tunable_widgets['calibrate_probabilities']
        calibrate_prob_widget.uncheck()

        if calibrate_prob_widget.get_value():
            calibrate_prob_widget.set_value(value=False)
            messagebox.showwarning(
                parent=self.window,
                title='Incorrect Configuration',
                message=f'The outputs of Neural Network cannot be calibrated. This option is set to False.'
            )

        super()._train()

    def _get_dialog_result(self):
        return None
