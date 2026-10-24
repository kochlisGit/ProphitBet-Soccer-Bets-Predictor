import pandas as pd
from typing import List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.leagues.league import LeagueConfig
from src.core.models.classifiers.lda import DiscriminantAnalysis
from src.core.preprocessing.utils import NormalizerType, SamplerType
from src.gui.components.dialogs.models.train.classifiers.classifier import ClassifierDialog
from src.gui.widgets import AnimatedToggleSwitch, TunableWidget


class DiscriminantAnalysisTrainerDialog(ClassifierDialog):
    """ Training dialog for LDA/QDA classifier.

        Attributes
        ----------
        :param database: (ModelDatabase) active league's model database instance.
        :param league_config: (LeagueConfig) configuration of the league, used to fetch the league's id.
        :param df: (pd.DataFrame) dataframe with the leagued data.
        :param parent: (QWidget | None) optional parent of the dialog.
    """

    def __init__(self, database: ModelDatabase, league_config: LeagueConfig, df: pd.DataFrame, parent: Optional[QWidget] = None):
        self._value_getters = {}
        super().__init__(model_cls=DiscriminantAnalysis, database=database, league_config=league_config, df=df, parent=parent)

    def _add_hyperparameters(self, frame: QFrame, layout: QVBoxLayout) -> List[TunableWidget]:
        """ Add the LDA/QDA hyperparameter controls. """

        normalizer_combo = QComboBox(frame)
        normalizer_combo.setFixedWidth(200)
        normalizer_combo.setMinimumHeight(32)
        normalizer_combo.addItem('None', userData=None)
        for normalizer in NormalizerType:
            normalizer_combo.addItem(normalizer.value, userData=normalizer)
        normalizer_combo.setCurrentIndex(3)
        normalizer_combo.setToolTip('Data normalization method. It usually increases the accuracy of the algorithm.')

        sampler_combo = QComboBox(frame)
        sampler_combo.setFixedWidth(200)
        sampler_combo.setMinimumHeight(32)
        sampler_combo.addItem('None', userData=None)
        for sampler in SamplerType:
            sampler_combo.addItem(sampler.value, userData=sampler)
        sampler_combo.setToolTip(
            'Data sampling method (oversampling/undersampling). It is effective when imbalanced classes exist.'
        )

        calibration_toggle = AnimatedToggleSwitch(checked=True, parent=frame)
        calibration_toggle.setToolTip(
            'Whether to calibrate output probabilities. A well-calibrated classifier is usually more robust.'
        )

        oas_toggle = AnimatedToggleSwitch(checked=False, parent=frame)
        oas_toggle.setToolTip('Whether to use the Oracle Approximating Shrinkage covariance estimator.')

        decision_boundary_combo = QComboBox(frame)
        decision_boundary_combo.setFixedWidth(200)
        decision_boundary_combo.setMinimumHeight(32)
        decision_boundary_combo.addItem('Linear', userData='linear')
        decision_boundary_combo.addItem('Quadratic', userData='quadratic')
        decision_boundary_combo.setToolTip('Decision boundary: set "linear" for line and "quadratic" for curve.')

        reg_param_control = QWidget(frame)
        reg_param_layout = QHBoxLayout(reg_param_control)
        reg_param_layout.setContentsMargins(0, 0, 0, 0)
        reg_param_layout.setSpacing(10)

        reg_param_slider = QSlider(Qt.Orientation.Horizontal, reg_param_control)
        reg_param_slider.setRange(0, 10)
        reg_param_slider.setSingleStep(1)
        reg_param_slider.setPageStep(1)
        reg_param_slider.setTickInterval(1)
        reg_param_slider.setValue(0)
        reg_param_slider.setToolTip('Regularization parameter, which helps discriminant analysis algorithm to converge.')

        reg_param_value_label = QLabel('0.001', reg_param_control)
        reg_param_value_label.setMinimumWidth(42)
        reg_param_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reg_param_layout.addWidget(reg_param_slider, 1)
        reg_param_layout.addWidget(reg_param_value_label)
        reg_param_control.setFixedWidth(200)
        reg_param_control.setMinimumHeight(32)

        def get_reg_param() -> float:
            return round(min(0.001 + reg_param_slider.value() * 0.01, 0.1), 3)

        def update_reg_param_label(_value: int) -> None:
            reg_param_value_label.setText(f'{get_reg_param():.3f}')

        reg_param_slider.valueChanged.connect(update_reg_param_label)

        hyperparameter_widgets = [
            TunableWidget(
                name='Normalization Method',
                widget_id='normalizer',
                checked=False,
                widget=normalizer_combo,
                parent=frame
            ),
            TunableWidget(
                name='Sampling Method',
                widget_id='sampler',
                checked=False,
                widget=sampler_combo,
                parent=frame
            ),
            TunableWidget(
                name='Probability Calibration',
                widget_id='calibrate_probabilities',
                checked=False,
                widget=calibration_toggle,
                parent=frame
            ),
            TunableWidget(
                name='Oracle Approximating Shrinkage',
                widget_id='oas',
                checked=False,
                widget=oas_toggle,
                parent=frame
            ),
            TunableWidget(
                name='Decision Boundary',
                widget_id='decision_boundary',
                checked=False,
                widget=decision_boundary_combo,
                parent=frame
            ),
            TunableWidget(
                name='Regularization',
                widget_id='reg_param',
                checked=False,
                widget=reg_param_control,
                parent=frame
            )
        ]

        self._value_getters.update({
            'normalizer': normalizer_combo.currentData,
            'sampler': sampler_combo.currentData,
            'calibrate_probabilities': calibration_toggle.isChecked,
            'oas': oas_toggle.isChecked,
            'decision_boundary': decision_boundary_combo.currentData,
            'reg_param': get_reg_param,
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
        """ Return the current value of a LDA/QDA hyperparameter. """

        try:
            getter = self._value_getters[name]
        except KeyError as error:
            raise KeyError(f'Unknown LDA/QDA hyperparameter: {name}') from error

        return getter()
