import sys
from typing import Callable, List, Optional, Tuple
import pandas as pd
from PySide6.QtCore import Qt
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
from src.core.database.agent import AgentDatabase
from src.core.leagues.league import LeagueConfig
from src.core.preprocessing.utils import NormalizerType
from src.gui.components.dialogs.models.train.agents.agent import AgentDialog
from src.gui.components.dialogs.tasks import ThreadTask
from src.gui.widgets import AnimatedToggleSwitch, TunableWidget


def _import_ppo_agent():
    """Import Stable Baselines and the PPO agent in the GUI process."""

    from src.core.agents.policies.ppo import PPOAgent
    return PPOAgent


class PPOTrainerDialog(AgentDialog):
    """ Training dialog for the Proximal Policy Optimization agent. """

    def __init__(self, database: AgentDatabase, league_config: LeagueConfig, df: pd.DataFrame, parent: Optional[QWidget] = None):
        self._value_getters = {}

        # Import Stable Baselines only when this dialog is first opened. This
        # must use an in-process thread so the imported modules remain available.
        if 'src.core.agents.policies.ppo' not in sys.modules:
            import_task = ThreadTask(
                _import_ppo_agent,
                title='Loading PPO Agent',
                description='Initializing Stable Baselines. This is required only the first time.',
                parent=parent,
            )
            PPOAgent = import_task.run()
            import_task.deleteLater()
            if PPOAgent is None:
                raise RuntimeError('The PPO agent could not be initialized.')
        else:
            from src.core.agents.policies.ppo import PPOAgent

        super().__init__(agent_cls=PPOAgent, database=database, league_config=league_config, df=df, parent=parent)

    def _add_hyperparameters(self, frame: QFrame, layout: QVBoxLayout) -> List[TunableWidget]:
        """Add the PPO hyperparameter controls."""

        def create_slider_control(slider: QSlider, default_text: str, tooltip: str, formatter: Callable[[], str]) -> QWidget:
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

        def create_int_slider(
                minimum: int,
                maximum: int,
                step: int,
                default: int,
                tooltip: str,
        ) -> Tuple[QWidget, QSlider]:
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

            control = create_slider_control(
                slider=slider,
                default_text=str(default),
                tooltip=tooltip,
                formatter=lambda: update_value(slider.value()),
            )
            return control, slider

        def create_float_slider(
                minimum: float,
                maximum: float,
                step: float,
                default: float,
                decimals: int,
                tooltip: str,
        ) -> Tuple[QWidget, Callable[[], float]]:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, round((maximum - minimum) / step))
            slider.setSingleStep(1)
            slider.setPageStep(1)
            slider.setTickInterval(1)
            slider.setValue(round((default - minimum) / step))

            def get_value() -> float:
                return round(minimum + slider.value() * step, decimals)

            control = create_slider_control(
                slider=slider,
                default_text=f'{default:.{decimals}f}',
                tooltip=tooltip,
                formatter=lambda: f'{get_value():.{decimals}f}',
            )
            return control, get_value

        normalizer_combo = QComboBox(frame)
        normalizer_combo.setFixedWidth(200)
        normalizer_combo.setMinimumHeight(32)
        normalizer_combo.addItem('None', userData=None)
        for normalizer in NormalizerType:
            normalizer_combo.addItem(normalizer.value, userData=normalizer)
        normalizer_combo.setCurrentIndex(normalizer_combo.findData(NormalizerType.STANDARD))
        normalizer_combo.setToolTip('Input-data normalization method.')

        filters_control, filters_slider = create_int_slider(
            8, 64, 8, 16, 'Number of convolutional filters in the feature extractor.'
        )
        features_dim_control, features_dim_slider = create_int_slider(
            64, 256, 64, 256, 'Output dimension of the feature extractor.'
        )

        hidden_layers_spinbox = QSpinBox(frame)
        hidden_layers_spinbox.setRange(1, 2)
        hidden_layers_spinbox.setValue(1)
        hidden_layers_spinbox.setFixedWidth(100)
        hidden_layers_spinbox.setMinimumHeight(32)
        hidden_layers_spinbox.setToolTip('Number of hidden layers in the policy and value networks.')

        hidden_units_control, hidden_units_slider = create_int_slider(
            64, 256, 64, 256, 'Number of neurons in each policy and value hidden layer.'
        )

        activation_combo = QComboBox(frame)
        activation_combo.setFixedWidth(200)
        activation_combo.setMinimumHeight(32)
        for activation in ('tanh', 'relu', 'gelu'):
            activation_combo.addItem(activation.upper(), userData=activation)
        activation_combo.setCurrentIndex(2)
        activation_combo.setToolTip('Activation function used by the policy and value networks.')

        batch_size_combo = QComboBox(frame)
        batch_size_combo.setFixedWidth(200)
        batch_size_combo.setMinimumHeight(32)
        for batch_size in (32, 64, 128, 256):
            batch_size_combo.addItem(str(batch_size), userData=batch_size)
        batch_size_combo.setCurrentIndex(3)
        batch_size_combo.setToolTip('Number of rollout samples per optimizer update.')

        learning_rate_control, get_learning_rate = create_float_slider(
            0.0005, 0.02, 0.0005, 0.005, 4, 'Optimizer learning rate.'
        )
        train_steps_control, train_steps_slider = create_int_slider(
            50000, 10000000, 50000, 100000, 'Total number of environment training steps.'
        )
        eval_freq_control, eval_freq_slider = create_int_slider(
            250, 10000, 250, 1000, 'Number of training steps between evaluations.'
        )
        gamma_control, get_gamma = create_float_slider(
            0.700, 0.999, 0.001, 0.997, 3, 'Discount factor for future rewards.'
        )

        n_steps_combo = QComboBox(frame)
        n_steps_combo.setFixedWidth(200)
        n_steps_combo.setMinimumHeight(32)
        for n_steps in (256, 512, 1024, 2048):
            n_steps_combo.addItem(str(n_steps), userData=n_steps)
        n_steps_combo.setCurrentIndex(2)
        n_steps_combo.setToolTip('Number of environment steps collected before each policy update.')

        epochs_control, epochs_slider = create_int_slider(
            5, 30, 5, 15, 'Number of training epochs performed for each policy update.'
        )
        gae_lambda_control, get_gae_lambda = create_float_slider(
            0.90, 0.99, 0.01, 0.95, 2, 'Generalized Advantage Estimation factor.'
        )
        epsilon_clip_control, get_epsilon_clip = create_float_slider(
            0.10, 0.35, 0.05, 0.20, 2, 'Policy-ratio clipping factor.'
        )

        normalize_advantage_toggle = AnimatedToggleSwitch(checked=True, parent=frame)
        normalize_advantage_toggle.setToolTip('Normalize advantages before each policy update.')

        entropy_regularization_control, get_entropy_regularization = create_float_slider(
            0.000, 0.030, 0.001, 0.001, 3, 'Entropy regularization coefficient.'
        )

        hyperparameter_widgets = [
            TunableWidget('Normalization Method', 'normalizer', False, normalizer_combo, frame),
            TunableWidget('Convolutional Filters', 'filters', False, filters_control, frame),
            TunableWidget('Features Dimension', 'features_dim', False, features_dim_control, frame),
            TunableWidget('Hidden Layers', 'hidden_layers', False, hidden_layers_spinbox, frame),
            TunableWidget('Hidden Units', 'hidden_units', False, hidden_units_control, frame),
            TunableWidget('Activation', 'activation', False, activation_combo, frame),
            TunableWidget('Batch Size', 'batch_size', False, batch_size_combo, frame),
            TunableWidget('Learning Rate', 'learning_rate', False, learning_rate_control, frame),
            TunableWidget('Training Steps', 'train_steps', False, train_steps_control, frame),
            TunableWidget('Evaluation Frequency', 'eval_freq', False, eval_freq_control, frame),
            TunableWidget('Discount Factor', 'gamma', False, gamma_control, frame),
            TunableWidget('Rollout Steps', 'n_steps', False, n_steps_combo, frame),
            TunableWidget('Update Epochs', 'epochs', False, epochs_control, frame),
            TunableWidget('GAE Lambda', 'gae_lambda', False, gae_lambda_control, frame),
            TunableWidget('Epsilon Clip', 'epsilon_clip', False, epsilon_clip_control, frame),
            TunableWidget(
                'Normalize Advantage', 'normalize_advantage', False, normalize_advantage_toggle, frame
            ),
            TunableWidget(
                'Entropy Regularization',
                'entropy_regularization',
                False,
                entropy_regularization_control,
                frame,
            ),
        ]

        non_tunable_parameters = {'train_steps', 'eval_freq', 'n_steps'}
        for widget in hyperparameter_widgets:
            if widget.widget_id in non_tunable_parameters:
                widget.findChild(QCheckBox).setEnabled(False)

        self._value_getters.update({
            'normalizer': normalizer_combo.currentData,
            'filters': filters_slider.value,
            'features_dim': features_dim_slider.value,
            'hidden_layers': hidden_layers_spinbox.value,
            'hidden_units': hidden_units_slider.value,
            'activation': activation_combo.currentData,
            'batch_size': batch_size_combo.currentData,
            'learning_rate': get_learning_rate,
            'train_steps': train_steps_slider.value,
            'eval_freq': eval_freq_slider.value,
            'gamma': get_gamma,
            'n_steps': n_steps_combo.currentData,
            'epochs': epochs_slider.value,
            'gae_lambda': get_gae_lambda,
            'epsilon_clip': get_epsilon_clip,
            'normalize_advantage': normalize_advantage_toggle.isChecked,
            'entropy_regularization': get_entropy_regularization,
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
        """Return the current value of a PPO hyperparameter."""

        try:
            getter = self._value_getters[name]
        except KeyError as error:
            raise KeyError(f'Unknown PPO hyperparameter: {name}') from error

        return getter()
