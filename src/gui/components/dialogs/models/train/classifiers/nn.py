import sys
import pandas as pd
from typing import Callable, List, Optional, Tuple
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget
)
from src.core.database.model import ModelDatabase
from src.core.leagues.league import LeagueConfig
from src.core.preprocessing.utils import NormalizerType, SamplerType
from src.gui.components.dialogs.models.train.classifiers.classifier import ClassifierDialog
from src.gui.components.dialogs.tasks import ThreadTask
from src.gui.widgets import AnimatedToggleSwitch, TunableWidget


def _import_neural_network():
    """Import TensorFlow and the neural-network classifier in the GUI process."""

    from src.core.models.classifiers.neuralnets.nn import NeuralNetwork
    return NeuralNetwork


class NeuralNetworkTrainerDialog(ClassifierDialog):
    """ Training dialog for Neural Network classifier.

        Attributes
        ----------
        :param database: (ModelDatabase) active league's model database instance.
        :param league_config: (LeagueConfig) configuration of the league, used to fetch the league's id.
        :param df: (pd.DataFrame) dataframe with the leagued data.
        :param parent: (QWidget | None) optional parent of the dialog.
    """


    def __init__(self, database: ModelDatabase, league_config: LeagueConfig, df: pd.DataFrame, parent: Optional[QWidget] = None):
        self._value_getters = {}

        # Import TensorFlow only when this dialog is first opened. This must use
        # an in-process thread so the imported modules remain available here.
        if 'tensorflow' not in sys.modules:
            import_task = ThreadTask(
                _import_neural_network,
                title='Loading Neural Network',
                description='Initializing TensorFlow. This is required only the first time.',
                parent=parent,
            )
            NeuralNetwork = import_task.run()
            import_task.deleteLater()
            if NeuralNetwork is None:
                raise RuntimeError('TensorFlow could not be initialized.')
        else:
            from src.core.models.classifiers.neuralnets.nn import NeuralNetwork

        super().__init__(model_cls=NeuralNetwork, database=database, league_config=league_config, df=df, parent=parent)

        blocker = QSignalBlocker(self._fast_tuning_checkbox)
        self._fast_tuning_checkbox.setChecked(True)
        del blocker
        self._fast_tuning_checkbox.setEnabled(False)
        self._nested_cv_checkbox.setChecked(False)
        self._nested_cv_checkbox.setEnabled(False)

    def _add_hyperparameters(self, frame: QFrame, layout: QVBoxLayout) -> List[TunableWidget]:
        """ Add neural network controls. Verbose option is not available. """

        def create_slider_control(slider: QSlider, default_text: str, tooltip: str,
                                  formatter: Callable[[], str]) -> QWidget:
            control = QWidget(frame)
            control_layout = QHBoxLayout(control)
            control_layout.setContentsMargins(0, 0, 0, 0)
            control_layout.setSpacing(10)
            slider.setParent(control)
            slider.setToolTip(tooltip)

            value_label = QLabel(default_text, control)
            value_label.setMinimumWidth(48)
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            control_layout.addWidget(slider, 1)
            control_layout.addWidget(value_label)
            control.setFixedWidth(200)
            control.setMinimumHeight(32)
            slider.valueChanged.connect(lambda _value: value_label.setText(formatter()))
            return control

        def create_int_slider(minimum: int, maximum: int, step: int, default: int,
                              tooltip: str) -> Tuple[QWidget, QSlider]:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(minimum, maximum)
            slider.setSingleStep(step)
            slider.setPageStep(step)
            slider.setTickInterval(step)
            slider.setValue(default)

            def update_value(value: int) -> str:
                snapped_value = minimum + round((value - minimum) / step) * step
                snapped_value = min(maximum, max(minimum, snapped_value))
                if snapped_value != value:
                    slider.setValue(snapped_value)
                return str(slider.value())

            control = create_slider_control(slider, str(default), tooltip, lambda: update_value(slider.value()))
            return control, slider

        def create_float_slider(minimum: float, maximum: float, step: float, default: float,
                                decimals: int, tooltip: str) -> tuple[QWidget, Callable[[], float]]:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, round((maximum - minimum) / step))
            slider.setSingleStep(1)
            slider.setPageStep(1)
            slider.setTickInterval(1)
            slider.setValue(round((default - minimum) / step))

            def get_value() -> float:
                return round(minimum + slider.value() * step, decimals)

            control = create_slider_control(
                slider, f'{default:.{decimals}f}', tooltip, lambda: f'{get_value():.{decimals}f}'
            )
            return control, get_value

        normalizer_combo = QComboBox(frame)
        normalizer_combo.setFixedWidth(200)
        normalizer_combo.setMinimumHeight(32)
        normalizer_combo.addItem('None', userData=None)
        for normalizer in NormalizerType:
            normalizer_combo.addItem(normalizer.value, userData=normalizer)
        normalizer_combo.setCurrentIndex(3)

        sampler_combo = QComboBox(frame)
        sampler_combo.setFixedWidth(200)
        sampler_combo.setMinimumHeight(32)
        sampler_combo.addItem('None', userData=None)
        for sampler in SamplerType:
            sampler_combo.addItem(sampler.value, userData=sampler)

        hidden_layers_spinbox = QSpinBox(frame)
        hidden_layers_spinbox.setRange(1, 3)
        hidden_layers_spinbox.setValue(1)
        hidden_layers_spinbox.setFixedWidth(100)
        hidden_layers_spinbox.setMinimumHeight(32)

        hidden_units_control, hidden_units_slider = create_int_slider(
            64, 256, 32, 256, 'Number of neurons in each hidden layer.'
        )
        hidden_activation_combo = QComboBox(frame)
        hidden_activation_combo.setFixedWidth(200)
        hidden_activation_combo.setMinimumHeight(32)
        for activation in ('tanh', 'relu', 'gelu'):
            hidden_activation_combo.addItem(activation.upper(), userData=activation)
        hidden_activation_combo.setCurrentIndex(2)

        vsn_toggle = AnimatedToggleSwitch(checked=True, parent=frame)
        layer_normalization_toggle = AnimatedToggleSwitch(checked=True, parent=frame)
        batch_normalization_toggle = AnimatedToggleSwitch(checked=True, parent=frame)
        dropout_rate_control, get_dropout_rate = create_float_slider(
            0.0, 0.5, 0.1, 0.2, 1, 'Dropout regularization rate.'
        )
        odd_noise_control, get_odd_noise = create_float_slider(
            0.0, 0.2, 0.01, 0.1, 2, 'Standard deviation of noise applied to odds inputs.'
        )
        class_weight_toggle = AnimatedToggleSwitch(checked=True, parent=frame)

        optimizer_combo = QComboBox(frame)
        optimizer_combo.setFixedWidth(200)
        optimizer_combo.setMinimumHeight(32)
        for optimizer in ('adam', 'adabelief', 'adan', 'ranger25'):
            optimizer_combo.addItem(optimizer, userData=optimizer)

        lookahead_toggle = AnimatedToggleSwitch(checked=True, parent=frame)
        label_smoothing_control, get_label_smoothing = create_float_slider(
            0.0, 0.1, 0.01, 0.1, 2, 'Label-smoothing factor.'
        )
        learning_rate_control, get_learning_rate = create_float_slider(
            0.0005, 0.05, 0.0005, 0.001, 4, 'Initial optimizer learning rate.'
        )
        batch_size_control, batch_size_slider = create_int_slider(
            16, 128, 16, 16, 'Number of samples per optimizer update.'
        )
        epochs_control, epochs_slider = create_int_slider(
            30, 100, 1, 30, 'Maximum number of training epochs.'
        )
        early_stopping_control, early_stopping_slider = create_int_slider(
            10, 30, 1, 20, 'Epochs without improvement before early stopping.'
        )
        lr_decay_patience_control, lr_decay_patience_slider = create_int_slider(
            0, 30, 5, 20, 'Epochs without improvement before reducing the learning rate.'
        )
        lr_decay_factor_control, get_lr_decay_factor = create_float_slider(
            0.2, 0.5, 0.1, 0.2, 1, 'Learning-rate reduction factor.'
        )

        hyperparameter_widgets = [
            TunableWidget('Normalization Method', 'normalizer', False, normalizer_combo, frame),
            TunableWidget('Sampling Method', 'sampler', False, sampler_combo, frame),
            TunableWidget('Hidden Layers', 'hidden_layers', False, hidden_layers_spinbox, frame),
            TunableWidget('Hidden Units', 'hidden_units', False, hidden_units_control, frame),
            TunableWidget('Hidden Activation', 'hidden_activation', False, hidden_activation_combo, frame),
            TunableWidget('Variable Selection Network', 'vsn', False, vsn_toggle, frame),
            TunableWidget('Layer Normalization', 'layer_normalization', False, layer_normalization_toggle, frame),
            TunableWidget('Batch Normalization', 'batch_normalization', False, batch_normalization_toggle, frame),
            TunableWidget('Dropout Rate', 'dropout_rate', False, dropout_rate_control, frame),
            TunableWidget('Odds Noise', 'odd_noise_std', False, odd_noise_control, frame),
            TunableWidget('Class Weight', 'class_weight', False, class_weight_toggle, frame),
            TunableWidget('Optimizer', 'optimizer', False, optimizer_combo, frame),
            TunableWidget('Lookahead', 'lookahead', False, lookahead_toggle, frame),
            TunableWidget('Label Smoothing', 'label_smoothing', False, label_smoothing_control, frame),
            TunableWidget('Learning Rate', 'learning_rate', False, learning_rate_control, frame),
            TunableWidget('Batch Size', 'batch_size', False, batch_size_control, frame),
            TunableWidget('Epochs', 'epochs', False, epochs_control, frame),
            TunableWidget('Early Stopping Patience', 'early_stopping_patience', False, early_stopping_control, frame),
            TunableWidget('LR Decay Patience', 'lr_decay_patience', False, lr_decay_patience_control, frame),
            TunableWidget('LR Decay Factor', 'lr_decay_factor', False, lr_decay_factor_control, frame),
        ]

        for widget in hyperparameter_widgets:
            if widget.widget_id in ('epochs', 'early_stopping_patience'):
                widget.findChild(QCheckBox).setEnabled(False)

        self._value_getters.update({
            'normalizer': normalizer_combo.currentData,
            'sampler': sampler_combo.currentData,
            'hidden_layers': hidden_layers_spinbox.value,
            'hidden_units': hidden_units_slider.value,
            'hidden_activation': hidden_activation_combo.currentData,
            'vsn': vsn_toggle.isChecked,
            'layer_normalization': layer_normalization_toggle.isChecked,
            'batch_normalization': batch_normalization_toggle.isChecked,
            'dropout_rate': get_dropout_rate,
            'odd_noise_std': get_odd_noise,
            'class_weight': class_weight_toggle.isChecked,
            'optimizer': optimizer_combo.currentData,
            'lookahead': lookahead_toggle.isChecked,
            'label_smoothing': get_label_smoothing,
            'learning_rate': get_learning_rate,
            'batch_size': batch_size_slider.value,
            'epochs': epochs_slider.value,
            'early_stopping_patience': early_stopping_slider.value,
            'lr_decay_patience': lr_decay_patience_slider.value,
            'lr_decay_factor': get_lr_decay_factor,
        })

        controls_layout = QGridLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(28)
        controls_layout.setVerticalSpacing(14)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for index, hyperparameter_widget in enumerate(hyperparameter_widgets):
            controls_layout.addWidget(hyperparameter_widget, index // 2, index % 2)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        layout.addLayout(controls_layout)
        layout.addStretch(1)
        return hyperparameter_widgets

    def _get_value(self, name: str) -> object:
        """ Return the current value of a neural network hyperparameter. """

        try:
            getter = self._value_getters[name]
        except KeyError as error:
            raise KeyError(f'Unknown neural network hyperparameter: {name}') from error
        return getter()
