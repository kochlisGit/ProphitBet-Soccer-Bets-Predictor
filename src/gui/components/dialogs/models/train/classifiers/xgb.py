import pandas as pd
from typing import Callable, List, Optional, Tuple
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.leagues.league import LeagueConfig
from src.core.models.classifiers.xgb import XGBoost
from src.core.preprocessing.utils import SamplerType
from src.gui.components.dialogs.models.train.classifiers.classifier import ClassifierDialog
from src.gui.widgets import AnimatedToggleSwitch, TunableWidget


class XGBoostTrainerDialog(ClassifierDialog):
    """ Training dialog for the XGBoost classifier.

        Attributes
        ----------
        :param database: (ModelDatabase) active league's model database instance.
        :param league_config: (LeagueConfig) configuration of the league, used to fetch the league's id.
        :param df: (pd.DataFrame) dataframe with the leagued data.
        :param parent: (QWidget | None) optional parent of the dialog.
    """

    def __init__(self, database: ModelDatabase, league_config: LeagueConfig, df: pd.DataFrame, parent: Optional[QWidget] = None):
        self._value_getters = {}
        super().__init__(model_cls=XGBoost, database=database, league_config=league_config, df=df, parent=parent)

    def _add_hyperparameters(self, frame: QFrame, layout: QVBoxLayout) -> List[TunableWidget]:
        """ Add the XGBoost hyperparameter controls. """

        def create_slider_control(
                slider: QSlider,
                default_text: str,
                tooltip: str,
                formatter: Callable[[], str]
        ) -> QWidget:
            control = QWidget(frame)
            control_layout = QHBoxLayout(control)
            control_layout.setContentsMargins(0, 0, 0, 0)
            control_layout.setSpacing(10)

            slider.setParent(control)
            slider.setToolTip(tooltip)
            value_label = QLabel(default_text, control)
            value_label.setMinimumWidth(42)
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
                formatter=lambda: f'{get_value():.{decimals}f}'
            )
            return control, get_value

        sampler_combo = QComboBox(frame)
        sampler_combo.setFixedWidth(200)
        sampler_combo.setMinimumHeight(32)
        sampler_combo.addItem('None', userData=None)
        for sampler in SamplerType:
            sampler_combo.addItem(sampler.value, userData=sampler)
        sampler_combo.setToolTip('Data normalization method. It usually increases the accuracy of the algorithm.')

        calibration_toggle = AnimatedToggleSwitch(checked=True, parent=frame)
        calibration_toggle.setToolTip(
            'Whether to calibrate output probabilities. A well-calibrated classifier is usually more robust.'
        )

        n_estimators_control, n_estimators_slider = create_int_slider(
            50, 1000, 50, 100, 'Number of boosted decision trees.'
        )
        max_depth_control, max_depth_slider = create_int_slider(
            1, 15, 1, 1, 'Maximum tree depth.'
        )
        min_child_weight_control, min_child_weight_slider = create_int_slider(
            1, 20, 1, 1, 'Minimum sum of instance weights required in a child node.'
        )
        learning_rate_control, get_learning_rate = create_float_slider(
            0.005,
            0.3,
            0.005,
            0.3,
            3,
            'Gradient-descent learning rate, which helps algorithm converge faster.'
        )
        subsample_control, get_subsample = create_float_slider(
            0.6, 1.0, 0.2, 1.0, 1, 'Fraction of training rows sampled for each tree.'
        )
        colsample_bytree_control, get_colsample_bytree = create_float_slider(
            0.6, 1.0, 0.2, 1.0, 1, 'Fraction of features sampled for each tree.'
        )
        reg_lambda_control, get_reg_lambda = create_float_slider(
            0.1, 10.0, 0.1, 1.0, 1, 'L2 regularization coefficient.'
        )
        reg_alpha_control, get_reg_alpha = create_float_slider(
            0.0, 10.0, 0.1, 0.0, 1, 'L1 regularization coefficient.'
        )

        hyperparameter_widgets = [
            TunableWidget('Sampling Method', 'sampler', False, sampler_combo, frame),
            TunableWidget('Probability Calibration', 'calibrate_probabilities', False, calibration_toggle, frame),
            TunableWidget('Num. Estimators', 'n_estimators', False, n_estimators_control, frame),
            TunableWidget('Max. Depth', 'max_depth', False, max_depth_control, frame),
            TunableWidget('Min. Child Weight', 'min_child_weight', False, min_child_weight_control, frame),
            TunableWidget('Learning Rate', 'learning_rate', False, learning_rate_control, frame),
            TunableWidget('Subsample', 'subsample', False, subsample_control, frame),
            TunableWidget('Column Subsample', 'colsample_bytree', False, colsample_bytree_control, frame),
            TunableWidget('Lambda Regularization', 'reg_lambda', False, reg_lambda_control, frame),
            TunableWidget('Alpha Regularization', 'reg_alpha', False, reg_alpha_control, frame)
        ]

        self._value_getters.update({
            'sampler': sampler_combo.currentData,
            'calibrate_probabilities': calibration_toggle.isChecked,
            'n_estimators': n_estimators_slider.value,
            'max_depth': max_depth_slider.value,
            'min_child_weight': min_child_weight_slider.value,
            'learning_rate': get_learning_rate,
            'subsample': get_subsample,
            'colsample_bytree': get_colsample_bytree,
            'reg_lambda': get_reg_lambda,
            'reg_alpha': get_reg_alpha
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
        """ Return the current value of an XGBoost hyperparameter. """

        try:
            getter = self._value_getters[name]
        except KeyError as error:
            raise KeyError(f'Unknown XGBoost hyperparameter: {name}') from error

        return getter()
