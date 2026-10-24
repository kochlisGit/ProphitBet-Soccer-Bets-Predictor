import pandas as pd
from typing import List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFrame, QGridLayout, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.leagues.league import LeagueConfig
from src.core.models.classifiers.logistic import LogisticRegressor
from src.core.preprocessing.utils import NormalizerType, SamplerType
from src.gui.components.dialogs.models.train.classifiers.classifier import ClassifierDialog
from src.gui.widgets import AnimatedToggleSwitch, TunableWidget


class LogisticRegressionTrainerDialog(ClassifierDialog):
    """ Training dialog for logistic-regression classifier.

        Attributes
        ----------
        :param database: (ModelDatabase) active league's model database instance.
        :param league_config: (LeagueConfig) configuration of the league, used to fetch the league's id.
        :param df: (pd.DataFrame) dataframe with the leagued data.
        :param parent: (QWidget | None) optional parent of the dialog.
    """

    def __init__(self, database: ModelDatabase, league_config: LeagueConfig, df: pd.DataFrame, parent: Optional[QWidget] = None):
        self._value_getters = {}
        super().__init__(model_cls=LogisticRegressor, database=database, league_config=league_config, df=df, parent=parent)

    def _add_hyperparameters(self, frame: QFrame, layout: QVBoxLayout) -> List[TunableWidget]:
        """ Add the logistic-regression hyperparameter controls. """

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

        penalty_combo = QComboBox(frame)
        penalty_combo.setFixedWidth(200)
        penalty_combo.setMinimumHeight(32)
        for penalty in (None, 'l1', 'l2', 'elastic'):
            penalty_combo.addItem('None' if penalty is None else penalty, userData=penalty)
        penalty_combo.setToolTip(
            'Penalty method. It typically penalizes large weights (coefficients), making the classifier more robust.'
        )

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
                name='Penalty',
                widget_id='penalty',
                checked=False,
                widget=penalty_combo,
                parent=frame
            )
        ]

        self._value_getters.update({
            'normalizer': normalizer_combo.currentData,
            'sampler': sampler_combo.currentData,
            'calibrate_probabilities': calibration_toggle.isChecked,
            'penalty': penalty_combo.currentData,
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
        """ Return the current value of a logistic-regression hyperparameter. """

        try:
            getter = self._value_getters[name]
        except KeyError as error:
            raise KeyError(f'Unknown logistic-regression hyperparameter: {name}') from error

        return getter()
