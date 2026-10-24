import pandas as pd
from typing import List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.leagues.league import LeagueConfig
from src.core.models.classifiers.svm import SupportVectorMachine
from src.core.preprocessing.utils import NormalizerType, SamplerType
from src.gui.components.dialogs.models.train.classifiers.classifier import ClassifierDialog
from src.gui.widgets import AnimatedToggleSwitch, TunableWidget


class SupportVectorMachineTrainerDialog(ClassifierDialog):
    """ Training dialog for SVM classifier.

        Attributes
        ----------
        :param database: (ModelDatabase) active league's model database instance.
        :param league_config: (LeagueConfig) configuration of the league, used to fetch the league's id.
        :param df: (pd.DataFrame) dataframe with the leagued data.
        :param parent: (QWidget | None) optional parent of the dialog.
    """

    def __init__(self, database: ModelDatabase, league_config: LeagueConfig, df: pd.DataFrame, parent: Optional[QWidget] = None):
        self._value_getters = {}
        super().__init__(model_cls=SupportVectorMachine, database=database, league_config=league_config, df=df, parent=parent)

    def _add_hyperparameters(self, frame: QFrame, layout: QVBoxLayout) -> List[TunableWidget]:
        """ Add the SVM hyperparameter controls. """

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

        kernel_combo = QComboBox(frame)
        kernel_combo.setFixedWidth(200)
        kernel_combo.setMinimumHeight(32)
        kernel_combo.addItem('None', userData=None)
        kernel_combo.addItem('Linear', userData='linear')
        kernel_combo.addItem('RBF', userData='rbf')
        kernel_combo.addItem('Polynomial', userData='poly')
        kernel_combo.addItem('Sigmoid', userData='sigmoid')
        kernel_combo.setCurrentIndex(2)
        kernel_combo.setToolTip('SVM kernel function. None uses a linear SVM without a kernel.')

        degree_control = QWidget(frame)
        degree_layout = QHBoxLayout(degree_control)
        degree_layout.setContentsMargins(0, 0, 0, 0)
        degree_layout.setSpacing(10)

        degree_slider = QSlider(Qt.Orientation.Horizontal, degree_control)
        degree_slider.setRange(3, 6)
        degree_slider.setSingleStep(1)
        degree_slider.setPageStep(1)
        degree_slider.setTickInterval(1)
        degree_slider.setValue(3)
        degree_slider.setToolTip('Polynomial degree (for Polynomial kernels only). ')

        degree_value_label = QLabel('3', degree_control)
        degree_value_label.setMinimumWidth(28)
        degree_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        degree_layout.addWidget(degree_slider, 1)
        degree_layout.addWidget(degree_value_label)
        degree_control.setFixedWidth(200)
        degree_control.setMinimumHeight(32)
        degree_slider.valueChanged.connect(lambda value: degree_value_label.setText(str(value)))

        gamma_control = QWidget(frame)
        gamma_layout = QHBoxLayout(gamma_control)
        gamma_layout.setContentsMargins(0, 0, 0, 0)
        gamma_layout.setSpacing(10)

        gamma_slider = QSlider(Qt.Orientation.Horizontal, gamma_control)
        gamma_slider.setRange(1, 20)
        gamma_slider.setSingleStep(1)
        gamma_slider.setPageStep(1)
        gamma_slider.setTickInterval(1)
        gamma_slider.setValue(10)
        gamma_slider.setToolTip('Regularization coefficient, which helps SVM converge into a stable solution.')

        gamma_value_label = QLabel('1.0', gamma_control)
        gamma_value_label.setMinimumWidth(32)
        gamma_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gamma_layout.addWidget(gamma_slider, 1)
        gamma_layout.addWidget(gamma_value_label)
        gamma_control.setFixedWidth(200)
        gamma_control.setMinimumHeight(32)

        def get_gamma() -> float:
            return gamma_slider.value() / 10

        gamma_slider.valueChanged.connect(lambda _value: gamma_value_label.setText(f'{get_gamma():.1f}'))

        class_weight_toggle = AnimatedToggleSwitch(checked=True, parent=frame)
        class_weight_toggle.setToolTip('Whether to apply class weights. It is beneficial when imbalanced classes exist.')

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
                name='Kernel',
                widget_id='kernel',
                checked=False,
                widget=kernel_combo,
                parent=frame
            ),
            TunableWidget(
                name='Degree',
                widget_id='degree',
                checked=False,
                widget=degree_control,
                parent=frame
            ),
            TunableWidget(
                name='Gamma',
                widget_id='gamma',
                checked=False,
                widget=gamma_control,
                parent=frame
            ),
            TunableWidget(
                name='Class Weight',
                widget_id='class_weight',
                checked=False,
                widget=class_weight_toggle,
                parent=frame
            )
        ]

        self._value_getters.update({
            'normalizer': normalizer_combo.currentData,
            'sampler': sampler_combo.currentData,
            'kernel': kernel_combo.currentData,
            'degree': degree_slider.value,
            'gamma': get_gamma,
            'class_weight': class_weight_toggle.isChecked,
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
        """ Return the current value of a SVM hyperparameter. """

        try:
            getter = self._value_getters[name]
        except KeyError as error:
            raise KeyError(f'Unknown SVM hyperparameter: {name}') from error

        return getter()
