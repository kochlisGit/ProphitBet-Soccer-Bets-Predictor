import pandas as pd
from typing import List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.leagues.league import LeagueConfig
from src.core.models.classifiers.knn import KNearestNeighbors
from src.core.preprocessing.utils import NormalizerType, SamplerType
from src.gui.components.dialogs.models.train.classifiers.classifier import ClassifierDialog
from src.gui.widgets import AnimatedToggleSwitch, TunableWidget


class KNearestNeighborsTrainerDialog(ClassifierDialog):
    """ Training dialog for KNN classifier.

        Attributes
        ----------
        :param database: (ModelDatabase) active league's model database instance.
        :param league_config: (LeagueConfig) configuration of the league, used to fetch the league's id.
        :param df: (pd.DataFrame) dataframe with the leagued data.
        :param parent: (QWidget | None) optional parent of the dialog.
    """

    def __init__(self, database: ModelDatabase, league_config: LeagueConfig, df: pd.DataFrame, parent: Optional[QWidget] = None):
        self._value_getters = {}
        super().__init__(model_cls=KNearestNeighbors, database=database, league_config=league_config, df=df, parent=parent)

    def _add_hyperparameters(self, frame: QFrame, layout: QVBoxLayout) -> List[TunableWidget]:
        """ Add the KNN hyperparameter controls. """

        normalizer_combo = QComboBox(frame)
        normalizer_combo.setFixedWidth(200)
        normalizer_combo.setMinimumHeight(32)
        normalizer_combo.addItem('None', userData=None)
        for normalizer in NormalizerType:
            normalizer_combo.addItem(normalizer.value, userData=normalizer)
        normalizer_combo.setCurrentIndex(1)
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

        neighbors_control = QWidget(frame)
        neighbors_layout = QHBoxLayout(neighbors_control)
        neighbors_layout.setContentsMargins(0, 0, 0, 0)
        neighbors_layout.setSpacing(10)

        neighbors_slider = QSlider(Qt.Orientation.Horizontal, neighbors_control)
        neighbors_slider.setRange(3, 151)
        neighbors_slider.setSingleStep(2)
        neighbors_slider.setPageStep(2)
        neighbors_slider.setTickInterval(2)
        neighbors_slider.setValue(5)
        neighbors_slider.setToolTip('Number of nearest neighbors to vote for the sample\'s label.')

        neighbors_value_label = QLabel('3', neighbors_control)
        neighbors_value_label.setMinimumWidth(28)
        neighbors_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        neighbors_layout.addWidget(neighbors_slider, 1)
        neighbors_layout.addWidget(neighbors_value_label)
        neighbors_control.setFixedWidth(200)
        neighbors_control.setMinimumHeight(32)

        def update_neighbors(value: int) -> None:
            snapped_value = 3 + round((value - 3) / 2) * 2
            if snapped_value != value:
                neighbors_slider.setValue(snapped_value)
                return
            neighbors_value_label.setText(str(value))

        neighbors_slider.valueChanged.connect(update_neighbors)

        weights_combo = QComboBox(frame)
        weights_combo.setFixedWidth(200)
        weights_combo.setMinimumHeight(32)
        weights_combo.addItem('Uniform', userData='uniform')
        weights_combo.addItem('Distance', userData='distance')
        weights_combo.setToolTip('Whether every neighbor votes equally or their vote is weighted by distance.')

        distance_combo = QComboBox(frame)
        distance_combo.setFixedWidth(200)
        distance_combo.setMinimumHeight(32)
        distance_combo.addItem('Manhattan', userData=1)
        distance_combo.addItem('Euclidean', userData=2)
        distance_combo.setCurrentIndex(1)
        distance_combo.setToolTip('Choose Manhattan (p=1) or Euclidean (p=2) distance.')

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
                name='Number of Neighbors',
                widget_id='n_neighbors',
                checked=False,
                widget=neighbors_control,
                parent=frame
            ),
            TunableWidget(
                name='Weights',
                widget_id='weights',
                checked=False,
                widget=weights_combo,
                parent=frame
            ),
            TunableWidget(
                name='Distance Metric',
                widget_id='p',
                checked=False,
                widget=distance_combo,
                parent=frame
            )
        ]

        self._value_getters.update({
            'normalizer': normalizer_combo.currentData,
            'sampler': sampler_combo.currentData,
            'calibrate_probabilities': calibration_toggle.isChecked,
            'n_neighbors': neighbors_slider.value,
            'weights': weights_combo.currentData,
            'p': distance_combo.currentData,
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
        """ Return the current value of a KNN hyperparameter. """

        try:
            getter = self._value_getters[name]
        except KeyError as error:
            raise KeyError(f'Unknown KNN hyperparameter: {name}') from error

        return getter()
