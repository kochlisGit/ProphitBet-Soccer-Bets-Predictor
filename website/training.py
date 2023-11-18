import ast
import threading
from abc import abstractmethod

import pandas as pd
from flask_wtf import FlaskForm
from wtforms import BooleanField, FloatField, IntegerField, SelectField, StringField
from wtforms.validators import InputRequired

from database.repositories.model import ModelRepository
from models.model import Model
from models.scikit.rf import RandomForest
from models.tf.nn import FCNet
from preprocessing import training


class CustomTrainForm(FlaskForm):
    def __init__(
        self,
        model_repository: ModelRepository,
        league_name: str,
        matches_df: pd.DataFrame,
        one_hot: bool,
        random_seed: int,
    ):
        super().__init__()

        self._model_repository = model_repository
        self._league_name = league_name
        self._inputs, self._targets = training.preprocess_training_dataframe(
            matches_df=matches_df, one_hot=one_hot
        )

        self._model = self._construct_model(
            input_shape=self._inputs.shape[1:], random_seed=random_seed
        )

    @property
    def model(self) -> Model:
        return self._model

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
        task_thread = threading.Thread(
            target=self._train,
            args=( use_over_sampling, num_eval_samples),
        )
        task_thread.start()

    def _train(
        self, use_over_sampling: bool, num_eval_samples: int
    ):
        self._build_model()

        x_train, y_train, x_test, y_test = training.split_train_targets(
            inputs=self._inputs,
            targets=self._targets,
            num_eval_samples=num_eval_samples,
        )
        self._eval_metrics = self._model.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            use_over_sampling=use_over_sampling,
        )

        self._model_repository.store_model(
            model=self._model, league_name=self._league_name
        )

    def submit_training(self):
        validation_result = self._validate_form()

        if validation_result == "valid":
            self._train_fn(
                use_over_sampling=self.use_oversampling.data,
                num_eval_samples=self.evaluation_samples.data,
            )
            validation_result += ". Model training in background, check terminal"
        return validation_result


    def _dialog_result(self):
        pass


class CustomTrainNNForm(CustomTrainForm):
    n_epochs = IntegerField("Number of epochs", validators=[InputRequired()], default=50)
    batch_size = IntegerField("Batch size", validators=[InputRequired()], default=35)
    early_stop_patience = FloatField("Early stp patience", validators=[InputRequired()], default=25)
    lr_decay_rate = FloatField("Learning rate decay rate", validators=[InputRequired()], default=0.2)
    lr_decay_patience = IntegerField("Learning rate decay patience", validators=[InputRequired()], default=10)
    lr = FloatField("Learning rate", validators=[InputRequired()], default=0.0002)
    input_noise_range = FloatField("Input noise range", validators=[InputRequired()], default=0.5)
    hidden_layers = StringField("Hidden Layers sizes", validators=[InputRequired()],default="128,64,32")
    hidden_layers_activations = StringField("Layers' activation", validators=[InputRequired()],default="relu,relu,tanh")
    hidden_layers_batch_norm = StringField("Layers' batch norm", validators=[InputRequired()],default="True,True,False")
    hidden_layers_regularization = StringField("Layers' regularization", validators=[InputRequired()],default="l2,None,None")
    hidden_layers_dropout = StringField("Layers' dropout", validators=[InputRequired()],default="0.0,0.0,0.2")
    optimizer = SelectField("Optimizer", validators=[InputRequired()])
    use_oversampling = BooleanField("Use oversampling")
    evaluation_samples = IntegerField("Evaluation samples", validators=[InputRequired()], default=10)

    def __init__(
        self,
        model_repository: ModelRepository,
        league_name: str,
        matches_df: pd.DataFrame,
        random_seed: int,
    ):
        super().__init__(
            model_repository=model_repository,
            league_name=league_name,
            matches_df=matches_df,
            one_hot=True,
            random_seed=random_seed,
        )
        self.optimizer.choices = ["adam", "adamw", "yogi"]

    def _validate_form(self) -> str:
        try:
            epochs = int(self.n_epochs.data)
            if epochs <= 0:
                return (
                    f'"epochs" is expected to be a positive integer e > 0, got {epochs}'
                )
        except ValueError:
            return f'"epochs" is expected to be a positive integer, got {self.n_epochs.data}'

        try:
            batch_size = int(self.batch_size.data)
            if batch_size <= 0:
                return f'"batch_size" is expected to be a positive integer b > 0, got {batch_size}'
        except ValueError:
            return f'"batch_size" is expected to be a positive integer b > 0, got {self.batch_size.data}'

        try:
            early_stopping_epochs = int(self.early_stop_patience.data)
            if early_stopping_epochs < 0:
                return (
                    '"early_stopping_epochs" is expected to be zero or a positive integer early >= 0'
                    f", got {early_stopping_epochs}"
                )
        except ValueError:
            return (
                '"early_stopping_epochs" is expected to be zero or a positive integer early >= 0'
                f", got {self.early_stop_patience.data}"
            )

        try:
            learning_rate_decay_factor = float(
                self.lr_decay_rate.data
            )
            if learning_rate_decay_factor < 0 or learning_rate_decay_factor >= 1.0:
                return (
                    '"learning_rate_decay_factor" is expected to be zero or a positive float between [0.0, 1.0), '
                    f"got {learning_rate_decay_factor}"
                )
        except ValueError:
            return (
                '"learning_rate_decay_factor" is expected to be zero or a positive float between [0.0, 1.0), '
                f"got {self.lr_decay_rate.data}"
            )

        try:
            learning_rate_decay_epochs = int(self.lr_decay_patience.data)
            if learning_rate_decay_epochs < 0:
                return (
                    '"lr_decay_patience" is expected to be zero or a positive integer early >= 0'
                    f", got {learning_rate_decay_epochs}"
                )
        except ValueError:
            return (
                '"learning_rate_decay_epochs" is expected to be zero or a positive integer early >= 0'
                f", got {self.lr_decay_patience.data}"
            )

        try:
            learning_rate = float(self.lr.data)
            if learning_rate <= 0 or learning_rate >= 1.0:
                return (
                    '"learning_rate" is expected to be zero or a positive float between [0.0, 1.0), '
                    f"got {learning_rate}"
                )
        except ValueError:
            return (
                '"learning_rate" is expected to be zero or a positive float between [0.0, 1.0), '
                f"got {self.lr.data}"
            )

        try:
            noise_range = float(self.input_noise_range.data)
            if noise_range < 0.0 or noise_range >= 1.0:
                return (
                    '"noise_range" is expected to be zero or a positive float between [0.0, 1.0), '
                    f"got {noise_range}"
                )
        except ValueError:
            return (
                '"noise_range" is expected to be zero or a positive float between [0.0, 1.0), '
                f"got {self.input_noise_range.data}"
            )

        try:
            hidden_layers_eval = ast.literal_eval(
                self.hidden_layers.data.replace(" ", "")
            )
            hidden_layers = [layer for layer in hidden_layers_eval]
            if not (
                all([isinstance(units, int) for units in hidden_layers])
                and all([int(units) > 0 for units in hidden_layers])
            ):
                return (
                    '"hidden_layers" should be positive integers, separated by comma, '
                    f"got {hidden_layers}"
                )
        except Exception:
            return (
                '"hidden_layers" should be positive integers, separated by comma, '
                f"got {self.hidden_layers.data.replace(' ', '')}"
            )

        try:
            activations = (
                self.hidden_layers_activations.data
                .replace(" ", "")
                .replace("[", "")
                .replace("]", "")
                .split(",")
            )
            activations = [
                activation if activation != "None" else None
                for activation in activations
            ]
            if not all(
                [
                    activation is None
                    or activation == "relu"
                    or activation == "gelu"
                    or activation == "tanh"
                    for activation in activations
                ]
            ):
                return (
                    f'"activations" should be either None or tanh or relu or gelu, '
                    f'separated by comma, all lowercase except "N" of None, got {activations}"'
                )
        except ValueError:
            return (
                '"activations" should be either None or tanh or relu or gelu or tanh, '
                'separated by comma, all lowercase except "N" of None, '
                f"got {self.hidden_layers_activations.data.replace(' ', '').replace('[', '').replace(']', '')}"
            )

        if len(hidden_layers) != len(activations):
            return (
                "Expected number of activations to equal number of hidden_layers (1 activation per layer), "
                f"got Hidden Layers: {len(hidden_layers)}, Activations: {len(activations)}"
            )

        try:
            batch_normalization_eval = ast.literal_eval(
                self.hidden_layers_batch_norm.data.replace(" ", "")
            )
            batch_normalizations = [
                batch_norm for batch_norm in batch_normalization_eval
            ]
            if not all(
                [isinstance(batch_norm, bool) for batch_norm in batch_normalizations]
            ):
                return (
                    '"batch_normalizations" should be either None or True or False, '
                    f'separated by comma, all lowercase except "T or F", got {batch_normalizations}"'
                )
        except ValueError:
            return (
                '"batch_normalizations" should be either None or True or False, '
                'separated by comma, all lowercase except "T" or "F", '
                f"got {self.hidden_layers_batch_norm.data.replace(' ', '')}"
            )

        if len(hidden_layers) != len(batch_normalizations):
            return (
                "Expected number of batch_normalizations layers to equal number of hidden_layers "
                "(1 batch normalization per layer), got Hidden Layers: "
                f"{len(hidden_layers)}, Batch Normalizations: {len(batch_normalizations)}"
            )

        try:
            regularizations = (
                self.hidden_layers_regularization.data
                .replace(" ", "")
                .replace("[", "")
                .replace("]", "")
                .split(",")
            )
            regularizations = [
                reg if reg != "None" else None for reg in regularizations
            ]
            if not all(
                [
                    regularization is None
                    or regularization == "l1"
                    or regularization == "l2"
                    for regularization in regularizations
                ]
            ):
                return (
                    '"regularizations" should be either None or l1 or l2, '
                    f'separated by comma, all lowercase except "N" of None, got {regularizations}"'
                )
        except ValueError:
            return (
                '"regularizations" should be either None or l1 or l2, '
                'separated by comma, all lowercase except "N" of None, '
                f"got {self.hidden_layers_regularization.data.replace(' ', '').replace('[', '').replace(']', '')}"
            )

        if len(hidden_layers) != len(regularizations):
            return (
                "Expected number of regularizations layers to equal number of hidden_layers "
                "(1 regularization per layer), got Hidden Layers: "
                f"{len(hidden_layers)}, Regularizations: {len(regularizations)}"
            )

        try:
            dropouts_eval = ast.literal_eval(self.hidden_layers_dropout.data.replace(" ", ""))
            dropouts = [dropout for dropout in dropouts_eval]
            if not (
                all([isinstance(dropout, float) for dropout in dropouts])
                and all([0.0 <= float(dropout) < 1.0 for dropout in dropouts])
            ):
                return (
                    '"dropouts" should be zero or positive float between [0.0, 1.0), separated by comma, '
                    f'got {dropouts}"'
                )
        except ValueError:
            return (
                f'"dropouts" should be zero or positive float between [0.0, 1.0), separated by comma, '
                f"got {self.hidden_layers_dropout.data.replace(' ', '')}"
            )

        if len(hidden_layers) != len(dropouts):
            return (
                "Expected number of dropout layers to equal number of hidden layers (1 dropout per layer), "
                f"got Hidden Layers: {len(hidden_layers)}, Dropouts: {len(dropouts)}"
            )

        optimizer = self.optimizer.data
        if not (optimizer == "adam" or optimizer == "adamw" or optimizer == "yogi"):
            return (
                f"Expected optimizer to equal adam, or adamw or yogi, got {optimizer}"
            )

        return "valid"

    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        return FCNet(input_shape=input_shape, random_seed=random_seed)

    def _build_model(self):
        activations = (
            self.hidden_layers_activations.data
            .replace(" ", "")
            .replace("[", "")
            .replace("]", "")
            .split(",")
        )
        activations = [
            activation if activation != "None" else None for activation in activations
        ]
        regularizations = (
            self.hidden_layers_regularization.data
            .replace(" ", "")
            .replace("[", "")
            .replace("]", "")
            .split(",")
        )
        regularizations = [reg if reg != "None" else None for reg in regularizations]

        self._model.build_model(
            epochs=self.n_epochs.data,
            batch_size=int(self.batch_size.data),
            early_stopping_epochs=self.early_stop_patience.data,
            learning_rate_decay_factor=float(
                self.lr_decay_rate.data
            ),
            learning_rate_decay_epochs=self.lr_decay_patience.data,
            learning_rate=float(self.lr.data),
            noise_range=float(self.input_noise_range.data),
            hidden_layers=list(ast.literal_eval(self.hidden_layers.data)),
            batch_normalizations=list(
                ast.literal_eval(self.hidden_layers_batch_norm.data)
            ),
            activations=activations,
            regularizations=regularizations,
            dropouts=list(ast.literal_eval(self.hidden_layers_dropout.data)),
            optimizer=self.optimizer.data,
        )


class CustomTrainRFForm(CustomTrainForm):
    n_estimators = IntegerField("Number of estimators", validators=[InputRequired()], default=100)
    max_features = SelectField("Max features", validators=[InputRequired()])
    max_depth = SelectField("Max depth", validators=[InputRequired()])
    min_samples_leaf = IntegerField("Min sample leaf", validators=[InputRequired()], default=1)
    min_samples_split = IntegerField("Min sample split", validators=[InputRequired()], default=2)
    bootstrap = BooleanField("Bootstrap")
    class_weight = SelectField("Class weight", validators=[InputRequired()])
    calibrate = BooleanField("Calibrate")
    use_oversampling = BooleanField("Use oversampling")
    evaluation_samples = IntegerField("N evaluation samples", validators=[InputRequired()], default=50)

    def __init__(
        self,
        model_repository: ModelRepository,
        league_name: str,
        matches_df: pd.DataFrame,
        random_seed: int,
    ):
        super().__init__(
            model_repository=model_repository,
            league_name=league_name,
            matches_df=matches_df,
            one_hot=False,
            random_seed=random_seed,
        )
        self.class_weight.choices = ["None", "balanced", "balanced_subsample"]
        self.max_features.choices = ["sqrt", "log2"]
        self.max_depth.choices = ["None"] + [str(i) for i in range(10, 101, 10)]

    def _validate_form(self) -> str:
        try:
            estimators = self.n_estimators.data
            if estimators <= 0:
                return f'"n_estimators" is expected to be a positive integer e > 0, got {estimators}'
        except ValueError:
            return f'"n_estimators" is expected to be a positive integer, got {self.n_estimators.data}'

        max_features = self.max_features.data
        if not (max_features == "sqrt" or max_features == "log2"):
            return f"Expected max_features to equal sqrt or log2, got {max_features}"

        try:
            max_depth = self.max_depth.data
            if max_depth != "None" and int(max_depth) <= 0:
                return f'"max_depth" is expected to be None or a positive integer, got {max_depth}'
        except ValueError:
            return f'"max_depth" is expected to be None or a positive integer, got {self.max_depth.data}'

        try:
            min_samples_leaf = int(self.min_samples_leaf.data)
            if min_samples_leaf <= 0:
                return f'"min_samples_leaf" is expected to be a positive integer, got {min_samples_leaf}'
        except ValueError:
            return f'"min_samples_leaf" is expected to be a positive integer, got {self.min_samples_leaf.data}'

        try:
            min_samples_split = int(self.min_samples_split.data)
            if min_samples_split <= 0:
                return f'"min_samples_split" is expected to be a positive integer, got {min_samples_split}'
        except ValueError:
            return f'"min_samples_split" is expected to be a positive integer, got {self.min_samples_split.data}'

        class_weight = self.class_weight.data
        if not (
            class_weight == "None"
            or class_weight == "balanced"
            or class_weight == "balanced_subsample"
        ):
            return f"Expected class_weight to be None or equal balanced or balanced_subsample, got {class_weight}"

        return "valid"

    def _construct_model(self, input_shape: tuple, random_seed: int) -> Model:
        return RandomForest(input_shape=input_shape, random_seed=random_seed)

    def _build_model(self):
        max_depth = self.max_depth.data
        max_depth = None if max_depth == "None" else int(max_depth)

        class_weight = self.class_weight.data
        if class_weight == "None":
            class_weight = None

        self._model.build_model(
            n_estimators=self.n_estimators.data,
            max_features=self.max_features.data,
            max_depth=max_depth,
            min_samples_leaf=self.min_samples_leaf.data,
            min_samples_split=self.min_samples_split.data,
            bootstrap=self.bootstrap.data,
            class_weight=class_weight,
            is_calibrated=self.calibrate.data,
        )
