import ast
import threading
import pandas as pd
from abc import ABC, abstractmethod
from tkinter import messagebox, Scale, StringVar, IntVar, BooleanVar
from tkinter.ttk import Label, Button, Combobox, Checkbutton, Entry
from database.repositories.model import ModelRepository
from gui.dialogs.dialog import Dialog
from gui.dialogs.model.utils import display_eval_metrics
from gui.dialogs.task import TaskDialog
from gui.widgets.utils import create_tooltip_btn, validate_float_entry
from models.model import Model
from models.scikit.rf import RandomForest
from models.tf.nn import FCNet
from preprocessing import training


class CustomTrainDialog(Dialog, ABC):
    def __init__(
            self,
            root,
            title: str,
            window_size: dict,
            model_repository: ModelRepository,
            league_name: str,
            matches_df: pd.DataFrame,
            one_hot: bool,
            random_seed: int
    ):
        super().__init__(root=root, title=title, window_size=window_size)

        self._model_repository = model_repository
        self._league_name = league_name
        self._inputs, self._targets = training.preprocess_training_dataframe(matches_df=matches_df, one_hot=one_hot)

        self._model = self._construct_model(input_shape=self._inputs.shape[1:], random_seed=random_seed)
        self._eval_metrics = None

        self._use_over_sampling_var = BooleanVar(value=True)
        self._num_eval_samples_var = IntVar(value=50)
        self._train_btn = Button(self.window, text='Train', command=self._submit_training)

    @property
    def model(self) -> Model:
        return self._model

    @property
    def use_over_sampling_var(self) -> BooleanVar:
        return self._use_over_sampling_var

    @property
    def num_eval_samples_var(self) -> IntVar:
        return self._num_eval_samples_var

    @property
    def train_btn(self) -> Button:
        return self._train_btn

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _validate_form(self) -> str:
        pass

    @abstractmethod
    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        pass

    @abstractmethod
    def _build_model(self):
        pass

    def _train_fn(self, use_over_sampling: bool, num_eval_samples: int):
        validation_result = self._validate_form()

        if validation_result == 'valid':
            task_dialog = TaskDialog(self._window, self._title)
            task_thread = threading.Thread(target=self._train, args=(task_dialog, use_over_sampling, num_eval_samples))
            task_thread.start()
            task_dialog.open()
        else:
            messagebox.showerror('Form Validation Error', validation_result)

    def _train(self, task_dialog: TaskDialog, use_over_sampling: bool, num_eval_samples: int):
        self._build_model()

        x_train, y_train, x_test, y_test = training.split_train_targets(
            inputs=self._inputs,
            targets=self._targets,
            num_eval_samples=num_eval_samples
        )
        self._eval_metrics = self._model.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            use_over_sampling=use_over_sampling
        )

        self._model_repository.store_model(model=self._model, league_name=self._league_name)
        task_dialog.close()

    def _submit_training(self):
        if messagebox.askokcancel(
                'Training Confirmation',
                'Training has been submitted. You will not be able to train '
                'another model until this session finishes. Do you wish to continue?'
        ):
            self._train_btn['state'] = 'disabled'
            self._train_fn(
                use_over_sampling=self._use_over_sampling_var.get(),
                num_eval_samples=self._num_eval_samples_var.get()
            )
            self._train_btn['state'] = 'enabled'

            if self._eval_metrics is not None:
                display_eval_metrics(self._eval_metrics)
                self._eval_metrics = None

    def _dialog_result(self):
        pass


class TrainCustomNNDialog(CustomTrainDialog):
    def __init__(
            self,
            root,
            model_repository: ModelRepository,
            league_name: str,
            matches_df: pd.DataFrame,
            random_seed: int
    ):
        super().__init__(
            root=root,
            title='Neural Network Training',
            window_size={'width': 545, 'height': 810},
            model_repository=model_repository,
            league_name=league_name,
            matches_df=matches_df,
            one_hot=True,
            random_seed=random_seed
        )

        self._epochs_var = IntVar(value=50)
        self._batch_size_var = StringVar(value='32')
        self._early_stopping_epochs_var = IntVar(value=25)
        self._learning_rate_decay_factor_var = StringVar(value='0.2')
        self._learning_rate_decay_epochs_var = IntVar(value=10)
        self._learning_rate_var = StringVar(value='0.001')
        self._noise_range_var = StringVar(value='0.5')
        self._hidden_layers_var = StringVar(value='128, 64, 32')
        self._activations_var = StringVar(value='relu, relu, tanh')
        self._batch_normalizations_var = StringVar(value='True, True, False')
        self._regularizations_var = StringVar(value='l2, None, None')
        self._dropouts_var = StringVar(value='0.0, 0.0, 0.2')
        self._optimizer_var = StringVar(value='adam')

    def _initialize(self):
        validate_int = self.window.register(validate_float_entry)
        validate_float = self.window.register(validate_float_entry)

        Label(self.window, text='Epochs', font=('Arial', 10)).place(x=20, y=15)
        Label(self.window, text='Batch Size', font=('Arial', 10)).place(x=20, y=65)
        Label(self.window, text='Early Stopping Patience', font=('Arial', 10)).place(x=20, y=115)
        Label(self.window, text='Learning Rate Decay Rate', font=('Arial', 10)).place(x=20, y=165)
        Label(self.window, text='Learning Rate Decay Patience', font=('Arial', 10)).place(x=20, y=215)
        Label(self.window, text='Learning Rate', font=('Arial', 10)).place(x=20, y=265)
        Label(self.window, text='Input Noise Range', font=('Arial', 10)).place(x=20, y=315)
        Label(self.window, text='Hidden Layers', font=('Arial', 10)).place(x=20, y=365)
        Label(self.window, text='Hidden Layer Activations', font=('Arial', 10)).place(x=20, y=415)
        Label(self.window, text='Hidden Layer Batch Normalizations', font=('Arial', 10)).place(x=20, y=465)
        Label(self.window, text='Hidden Layer Regularizations', font=('Arial', 10)).place(x=20, y=515)
        Label(self.window, text='Hidden Layer Dropout Rates', font=('Arial', 10)).place(x=20, y=565)
        Label(self.window, text='Optimizer', font=('Arial', 10)).place(x=20, y=615)
        Label(self.window, text='Use Input Over-Sampling', font=('Arial', 10)).place(x=20, y=665)
        Label(self.window, text='Evaluation Samples', font=('Arial', 10)).place(x=20, y=715)

        create_tooltip_btn(
            root=self.window, x=250, y=15,
            text='Number of training epochs. Should be positive integer, usually (50-150)'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=65,
            text='Training batch size. Should be positive integer, usually (16-128)'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=115,
            text='Number early stopping epochs to wait before stopping training,'
                 '\nif validation loss does not improve. Should be positive integer or zero,'
                 '\n usually (25-50). Set it to 0 to disable Early Stopping mechanism'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=165,
            text='Learning rate decay factor to reduce learning rate,'
                 '\nif validation loss does not improve. Should be positive float or zero between '
                 '\n0.0 and 1.0, usually (0.1-0.4). Set it to 0 to disable Learning Rate Decay'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=215,
            text='Number of epochs to wait for validation loss improvement,'
                 '\nbefore reducing learning rate. Should be positive integer or zero, usually (5-15).'
                 '\nSet it to 0 to disable Learning Rate Decay'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=265,
            text='Initial learning rate of optimizer. Should be a positive float between '
                 '\n0.0 and 1.0, usually (0.0005-0.01)'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=315,
            text='Noise amount to add to data at the beginning of each epoch. '
                 '\nShould be positive float or zero between 0.0 and 1.0, usually (0.0-0.5). '
                 '\nSet it to 0 to disable Input Noise'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=365,
            text='Number of Hidden Layers & Hidden Units per Layer (e.g. 64, 128, 128, 64)'
                 '\nor [256, 256, 128]. Usually, (1-4) layers will be fine with (16-256) units per layer.'
                 '\nNumber of units per layer should be separated by comma (,).'
                 '\nUnits should be positive integers'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=415,
            text='Activation function per hidden layer (e.g. relu, relu, tanh or [gelu, gelu].'
                 '\nAvailable activation functions are: (None, tanh, relu, gelu).'
                 '\nActivation per layer is separated by comma (,).'
                 '\n"None" requires capital N, the rest should be lowercase'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=465,
            text='Batch normalization per layer (e.g. True, True, False) or [True, True, False].'
                 '\nAvailable options are: (True, False). Batch Normalization per layer'
                 '\nis separated by comma (,). Both True & False require Capital T and F respectively'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=515,
            text='Regularization per layer (e.g. l2, None, None, l1) or [l1, l1, None].'
                 '\nAvailable options are: (None, l2, l1). Regularization per layer'
                 '\nis separated by comma (,). "None" requires capital N, the rest should be lowercase'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=565,
            text='Dropout rate per layer (e.g. 0.2, 0.2, 0.4) or [0.1, 0.5].'
                 '\nUsually, a good value is (0.2-0.4). Dropout rate should be positive '
                 '\nfloat or zero between 0.0 and 1.0.'
                 '\nIt is best to set all Batch Normalizations to False if using Dropout rate.'
                 '\nDropout rate per layer is separated by comma (,)'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=615,
            text='Optimizer for training neural network.'
                 '\nAvailable options are (adam, adamw, yogi).'
                 '\nAll letters should be lowercase'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=665,
            text='Use over-sampling to generate synthetic data for minority classes'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=715,
            text='Number of evaluation samples to exclude from training\nand use them as evaluation samples.'
        )

        Scale(
            self.window, from_=1, to=251, tickinterval=50, orient='horizontal', length=220, variable=self._epochs_var
        ).place(x=300, y=1)

        batch_size_cb = Combobox(
            self.window, width=10, font=('Arial', 10),
            validate='key', validatecommand=(validate_int, '%P'), textvariable=self._batch_size_var
        )
        batch_size_cb['values'] = ['8', '16', '32', '64', '128']
        batch_size_cb.current(2)
        batch_size_cb.place(x=300, y=65)

        Scale(
            self.window, from_=0, to=50, tickinterval=10, orient='horizontal',
            length=150, variable=self._early_stopping_epochs_var
        ).place(x=300, y=100)

        Entry(
            self.window, width=10, font=('Arial', 10),
            validate='key', validatecommand=(validate_float, '%P'), textvariable=self._learning_rate_decay_factor_var
        ).place(x=300, y=165)

        Scale(
            self.window, from_=0, to=30, tickinterval=10, orient='horizontal',
            length=100, variable=self._learning_rate_decay_epochs_var
        ).place(x=300, y=200)

        Entry(
            self.window, width=10, font=('Arial', 10),
            validate='key', validatecommand=(validate_float, '%P'), textvariable=self._learning_rate_var
        ).place(x=300, y=265)

        Entry(
            self.window, width=10, font=('Arial', 10),
            validate='key', validatecommand=(validate_float, '%P'), textvariable=self._noise_range_var
        ).place(x=300, y=315)

        Entry(
            self.window, width=20, font=('Arial', 10), textvariable=self._hidden_layers_var
        ).place(x=300, y=365)

        Entry(
            self.window, width=20, font=('Arial', 10), textvariable=self._activations_var
        ).place(x=300, y=415)

        Entry(
            self.window, width=20, font=('Arial', 10), textvariable=self._batch_normalizations_var
        ).place(x=300, y=465)

        Entry(
            self.window, width=20, font=('Arial', 10),  textvariable=self._regularizations_var
        ).place(x=300, y=515)

        Entry(
            self.window, width=20, font=('Arial', 10), textvariable=self._dropouts_var
        ).place(x=300, y=565)

        optimizer_size_cb = Combobox(
            self.window, width=10, font=('Arial', 10), state='readonly', textvariable=self._optimizer_var
        )
        optimizer_size_cb['values'] = ['adam', 'adamw', 'yogi']
        optimizer_size_cb.current(0)
        optimizer_size_cb.place(x=300, y=615)

        Checkbutton(
            self.window, onvalue=True, offvalue=False, text='Use Over-Sampling',
            variable=self.use_over_sampling_var
        ).place(x=300, y=665)

        Scale(
            self.window, from_=0, to=250, tickinterval=50, orient='horizontal',
            length=220, variable=self.num_eval_samples_var
        ).place(x=300, y=700)

        self.train_btn.place(x=220, y=770)

    def _validate_form(self) -> str:
        try:
            epochs = int(self._epochs_var.get())
            if epochs <= 0:
                return f'"epochs" is expected to be a positive integer e > 0, got {epochs}'
        except ValueError:
            return f'"epochs" is expected to be a positive integer, got {self._epochs_var.get()}'

        try:
            batch_size = int(self._batch_size_var.get())
            if batch_size <= 0:
                return f'"batch_size" is expected to be a positive integer b > 0, got {batch_size}'
        except ValueError:
            return f'"batch_size" is expected to be a positive integer b > 0, got {self._batch_size_var.get()}'

        try:
            early_stopping_epochs = int(self._early_stopping_epochs_var.get())
            if early_stopping_epochs < 0:
                return '"early_stopping_epochs" is expected to be zero or a positive integer early >= 0' \
                       f', got {early_stopping_epochs}'
        except ValueError:
            return '"early_stopping_epochs" is expected to be zero or a positive integer early >= 0' \
                   f', got {self._early_stopping_epochs_var.get()}'

        try:
            learning_rate_decay_factor = float(self._learning_rate_decay_factor_var.get())
            if learning_rate_decay_factor < 0 or learning_rate_decay_factor >= 1.0:
                return '"learning_rate_decay_factor" is expected to be zero or a positive float between [0.0, 1.0), ' \
                       f'got {learning_rate_decay_factor}'
        except ValueError:
            return '"learning_rate_decay_factor" is expected to be zero or a positive float between [0.0, 1.0), ' \
                   f'got {self._learning_rate_decay_factor_var.get()}'

        try:
            learning_rate_decay_epochs = int(self._learning_rate_decay_epochs_var.get())
            if learning_rate_decay_epochs < 0:
                return '"learning_rate_decay_epochs" is expected to be zero or a positive integer early >= 0' \
                       f', got {learning_rate_decay_epochs}'
        except ValueError:
            return '"learning_rate_decay_epochs" is expected to be zero or a positive integer early >= 0' \
                   f', got {self._learning_rate_decay_epochs_var.get()}'

        try:
            learning_rate = float(self._learning_rate_var.get())
            if learning_rate <= 0 or learning_rate >= 1.0:
                return '"learning_rate" is expected to be zero or a positive float between [0.0, 1.0), ' \
                       f'got {learning_rate}'
        except ValueError:
            return '"learning_rate" is expected to be zero or a positive float between [0.0, 1.0), ' \
                   f'got {self._learning_rate_var.get()}'

        try:
            noise_range = float(self._noise_range_var.get())
            if noise_range < 0.0 or noise_range >= 1.0:
                return '"noise_range" is expected to be zero or a positive float between [0.0, 1.0), ' \
                       f'got {noise_range}'
        except ValueError:
            return '"noise_range" is expected to be zero or a positive float between [0.0, 1.0), ' \
                   f'got {self._noise_range_var.get()}'

        try:
            hidden_layers_eval = ast.literal_eval(self._hidden_layers_var.get().replace(' ', ''))
            hidden_layers = [layer for layer in hidden_layers_eval]
            if not (
                    all([isinstance(units, int) for units in hidden_layers]) and
                    all([int(units) > 0 for units in hidden_layers])
            ):
                return '"hidden_layers" should be positive integers, separated by comma, ' \
                       f"got {hidden_layers}"
        except ValueError:
            return '"hidden_layers" should be positive integers, separated by comma, ' \
                   f"got {self._hidden_layers_var.get().replace(' ', '')}"

        try:
            activations = self._activations_var.get().replace(' ', '').replace('[', '').replace(']', '').split(',')
            activations = [activation if activation != 'None' else None for activation in activations]
            if not all([activation is None or activation == 'relu' or activation == 'gelu' or activation == 'tanh'
                        for activation in activations]):
                return f'"activations" should be either None or tanh or relu or gelu, ' \
                       f'separated by comma, all lowercase except "N" of None, got {activations}"'
        except ValueError:
            return '"activations" should be either None or tanh or relu or gelu or tanh, ' \
                   'separated by comma, all lowercase except "N" of None, ' \
                   f"got {self._activations_var.get().replace(' ', '').replace('[', '').replace(']', '')}"

        if len(hidden_layers) != len(activations):
            return 'Expected number of activations to equal number of hidden_layers (1 activation per layer), ' \
                   f'got Hidden Layers: {len(hidden_layers)}, Activations: {len(activations)}'

        try:
            batch_normalization_eval = ast.literal_eval(self._batch_normalizations_var.get().replace(' ', ''))
            batch_normalizations = [batch_norm for batch_norm in batch_normalization_eval]
            if not all([isinstance(batch_norm, bool) for batch_norm in batch_normalizations]):
                return '"batch_normalizations" should be either None or True or False, ' \
                       f'separated by comma, all lowercase except "T or F", got {batch_normalizations}"'
        except ValueError:
            return '"batch_normalizations" should be either None or True or False, ' \
                   'separated by comma, all lowercase except "T" or "F", ' \
                   f"got {self._batch_normalizations_var.get().replace(' ', '')}"

        if len(hidden_layers) != len(batch_normalizations):
            return 'Expected number of batch_normalizations layers to equal number of hidden_layers ' \
                   '(1 batch normalization per layer), got Hidden Layers: ' \
                   f'{len(hidden_layers)}, Batch Normalizations: {len(batch_normalizations)}'

        try:
            regularizations = self._regularizations_var.get().replace(
                ' ', '').replace('[', '').replace(']', '').split(',')
            regularizations = [reg if reg != 'None' else None for reg in regularizations]
            if not all(
                    [regularization is None or regularization == 'l1' or regularization == 'l2'
                     for regularization in regularizations]
            ):
                return '"regularizations" should be either None or l1 or l2, ' \
                       f'separated by comma, all lowercase except "N" of None, got {regularizations}"'
        except ValueError:
            return '"regularizations" should be either None or l1 or l2, ' \
                   'separated by comma, all lowercase except "N" of None, ' \
                   f"got {self._regularizations_var.get().replace(' ', '').replace('[', '').replace(']', '')}"

        if len(hidden_layers) != len(regularizations):
            return 'Expected number of regularizations layers to equal number of hidden_layers ' \
                   '(1 regularization per layer), got Hidden Layers: ' \
                   f'{len(hidden_layers)}, Regularizations: {len(regularizations)}'

        try:
            dropouts_eval = ast.literal_eval(self._dropouts_var.get().replace(' ', ''))
            dropouts = [dropout for dropout in dropouts_eval]
            if not (
                    all([isinstance(dropout, float) for dropout in dropouts]) and
                    all([0.0 <= float(dropout) < 1.0 for dropout in dropouts])
            ):
                return '"dropouts" should be zero or positive float between [0.0, 1.0), separated by comma, ' \
                       f'got {dropouts}"'
        except ValueError:
            return f'"dropouts" should be zero or positive float between [0.0, 1.0), separated by comma, ' \
                   f"got {self._dropouts_var.get().replace(' ', '')}"

        if len(hidden_layers) != len(dropouts):
            return 'Expected number of dropout layers to equal number of hidden layers (1 dropout per layer), ' \
                   f'got Hidden Layers: {len(hidden_layers)}, Dropouts: {len(dropouts)}'

        optimizer = self._optimizer_var.get()
        if not (optimizer == 'adam' or optimizer == 'adamw' or optimizer == 'yogi'):
            return f'Expected optimizer to equal adam, or adamw or yogi, got {optimizer}'

        return 'valid'

    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        return FCNet(input_shape=input_shape, random_seed=random_seed)

    def _build_model(self):
        activations = self._activations_var.get().replace(' ', '').replace('[', '').replace(']', '').split(',')
        activations = [activation if activation != 'None' else None for activation in activations]
        regularizations = self._regularizations_var.get().replace(' ', '').replace('[', '').replace(']', '').split(',')
        regularizations = [reg if reg != 'None' else None for reg in regularizations]

        self._model.build_model(
            epochs=self._epochs_var.get(),
            batch_size=int(self._batch_size_var.get()),
            early_stopping_epochs=self._early_stopping_epochs_var.get(),
            learning_rate_decay_factor=float(self._learning_rate_decay_factor_var.get()),
            learning_rate_decay_epochs=self._learning_rate_decay_epochs_var.get(),
            learning_rate=float(self._learning_rate_var.get()),
            noise_range=float(self._noise_range_var.get()),
            hidden_layers=list(ast.literal_eval(self._hidden_layers_var.get())),
            batch_normalizations=list(ast.literal_eval(self._batch_normalizations_var.get())),
            activations=activations,
            regularizations=regularizations,
            dropouts=list(ast.literal_eval(self._dropouts_var.get())),
            optimizer=self._optimizer_var.get()
        )


class TrainCustomRFDialog(CustomTrainDialog):
    def __init__(
            self,
            root,
            model_repository: ModelRepository,
            league_name: str,
            matches_df: pd.DataFrame,
            random_seed: int
    ):
        super().__init__(
            root=root,
            title='Random Forest Training',
            window_size={'width': 540, 'height': 570},
            model_repository=model_repository,
            league_name=league_name,
            matches_df=matches_df,
            one_hot=False,
            random_seed=random_seed
        )

        self._n_estimators_var = IntVar(value=100)
        self._max_features_var = StringVar(value='sqrt')
        self._max_depth_var = StringVar(value='None')
        self._min_samples_leaf_var = IntVar(value=1)
        self._min_samples_split_var = IntVar(value=2)
        self._bootstrap_var = BooleanVar(value=True)
        self._class_weight_var = StringVar(value=None)
        self._is_calibrated_var = BooleanVar(value=True)

    def _initialize(self):
        Label(self.window, text='Estimators', font=('Arial', 10)).place(x=20, y=15)
        Label(self.window, text='Max Features', font=('Arial', 10)).place(x=20, y=65)
        Label(self.window, text='Max Depth', font=('Arial', 10)).place(x=20, y=115)
        Label(self.window, text='Min Samples Leaf', font=('Arial', 10)).place(x=20, y=165)
        Label(self.window, text='Min Samples Split', font=('Arial', 10)).place(x=20, y=225)
        Label(self.window, text='Bootstrap', font=('Arial', 10)).place(x=20, y=275)
        Label(self.window, text='Class Weight', font=('Arial', 10)).place(x=20, y=325)
        Label(self.window, text='Calibrate', font=('Arial', 10)).place(x=20, y=375)
        Label(self.window, text='Use Input Over-Sampling', font=('Arial', 10)).place(x=20, y=425)
        Label(self.window, text='Evaluation Samples', font=('Arial', 10)).place(x=20, y=475)

        create_tooltip_btn(
            root=self.window, x=250, y=15,
            text='Number of trees in the forest. Should be positive integer, usually (100-200)'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=65,
            text='The number of features to consider when looking for the best tree split.\n'
                 'Should be positive sqrt or log2'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=115,
            text='The maximum depth of the tree. If None,'
                 '\nthen nodes are expanded until all leaves are pure.'
                 '\nShould be None or Positive Integer, usually None'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=165,
            text='The minimum number of samples required to be at a leaf node. '
                 '\nShould be positive integer, usually (1-5) '
        )
        create_tooltip_btn(
            root=self.window, x=250, y=225,
            text='The minimum number of samples required to split an internal node'
                 '\nShould be positive integer, usually (2-10)'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=275,
            text='Whether bootstrap samples are used when building trees. '
                 '\nIf False, the whole dataset is used to build each tree.'
                 '\nShould be set to either True or False'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=325,
            text='Weights associated with classes. If not given,'
                 '\nall classes are supposed to have weight one.'
                 '\nShould be balanced, balanced_subsample or None'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=375,
            text='Whether or not to use Probability calibration for imbalanced classes'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=425,
            text='Use over-sampling to generate synthetic data for minority classes'
        )
        create_tooltip_btn(
            root=self.window, x=250, y=475,
            text='Number of evaluation samples to exclude from training\nand use them as evaluation samples'
        )

        Scale(
            self.window, from_=1, to=500, tickinterval=100,
            orient='horizontal', length=220, variable=self._n_estimators_var
        ).place(x=300, y=1)

        max_features_cb = Combobox(
            self.window, width=10, font=('Arial', 10), state='readonly', textvariable=self._max_features_var
        )
        max_features_cb['values'] = ['sqrt', 'log2']
        max_features_cb.current(0)
        max_features_cb.place(x=300, y=65)

        max_depth_cb = Combobox(
            self.window, width=10, font=('Arial', 10), state='readonly', textvariable=self._max_depth_var
        )
        max_depth_cb['values'] = ['None'] + [str(i) for i in range(10, 101, 10)]
        max_depth_cb.current(0)
        max_depth_cb.place(x=300, y=115)

        Scale(
            self.window, from_=1, to=10, tickinterval=1,
            orient='horizontal', length=220, variable=self._min_samples_leaf_var
        ).place(x=300, y=150)

        Scale(
            self.window, from_=1, to=10, tickinterval=1,
            orient='horizontal', length=220, variable=self._min_samples_split_var
        ).place(x=300, y=210)

        Checkbutton(
            self.window, onvalue=True, offvalue=False, text='Bootstrap',
            variable=self._bootstrap_var
        ).place(x=300, y=275)

        class_weight_cb = Combobox(
            self.window, width=10, font=('Arial', 10), state='readonly', textvariable=self._class_weight_var
        )
        class_weight_cb['values'] = ['None', 'balanced', 'balanced_subsample']
        class_weight_cb.current(0)
        class_weight_cb.place(x=300, y=325)

        Checkbutton(
            self.window, onvalue=True, offvalue=False, text='Calibrate',
            variable=self._is_calibrated_var
        ).place(x=300, y=375)

        Checkbutton(
            self.window, onvalue=True, offvalue=False, text='Use Over-Sampling',
            variable=self.use_over_sampling_var
        ).place(x=300, y=425)

        Scale(
            self.window, from_=0, to=250, tickinterval=50, orient='horizontal',
            length=220, variable=self.num_eval_samples_var
        ).place(x=300, y=460)

        self.train_btn.place(x=220, y=530)

    def _validate_form(self) -> str:
        try:
            estimators = self._n_estimators_var.get()
            if estimators <= 0:
                return f'"n_estimators" is expected to be a positive integer e > 0, got {estimators}'
        except ValueError:
            return f'"n_estimators" is expected to be a positive integer, got {self._n_estimators_var.get()}'

        max_features = self._max_features_var.get()
        if not(max_features == 'sqrt' or max_features == 'log2'):
            return f'Expected max_features to equal sqrt or log2, got {max_features}'

        try:
            max_depth = self._max_depth_var.get()
            if max_depth != 'None' and int(max_depth) <= 0:
                return f'"max_depth" is expected to be None or a positive integer, got {max_depth}'
        except ValueError:
            return f'"max_depth" is expected to be None or a positive integer, got {self._max_depth_var.get()}'

        try:
            min_samples_leaf = int(self._min_samples_leaf_var.get())
            if min_samples_leaf <= 0:
                return f'"min_samples_leaf" is expected to be a positive integer, got {min_samples_leaf}'
        except ValueError:
            return f'"min_samples_leaf" is expected to be a positive integer, got {self._min_samples_leaf_var.get()}'

        try:
            min_samples_split = int(self._min_samples_split_var.get())
            if min_samples_split <= 0:
                return f'"min_samples_split" is expected to be a positive integer, got {min_samples_split}'
        except ValueError:
            return f'"min_samples_split" is expected to be a positive integer, got {self._min_samples_split_var.get()}'

        class_weight = self._class_weight_var.get()
        if not(class_weight == 'None' or class_weight == 'balanced' or class_weight == 'balanced_subsample'):
            return f'Expected class_weight to be None or equal balanced or balanced_subsample, got {class_weight}'

        return 'valid'

    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        return RandomForest(input_shape=input_shape, random_seed=random_seed)

    def _build_model(self):
        max_depth = self._max_depth_var.get()
        max_depth = None if max_depth == 'None' else int(max_depth)

        class_weight = self._class_weight_var.get()
        if class_weight == 'None':
            class_weight = None

        self._model.build_model(
            n_estimators=self._n_estimators_var.get(),
            max_features=self._max_features_var.get(),
            max_depth=max_depth,
            min_samples_leaf=self._min_samples_leaf_var.get(),
            min_samples_split=self._min_samples_split_var.get(),
            bootstrap=self._bootstrap_var.get(),
            class_weight=class_weight,
            is_calibrated=self._is_calibrated_var.get()
        )
