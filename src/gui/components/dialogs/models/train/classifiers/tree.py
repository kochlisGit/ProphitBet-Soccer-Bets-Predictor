import pandas as pd
from typing import List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.leagues.league import LeagueConfig
from src.core.models.classifiers.tree import DecisionTree
from src.core.preprocessing.utils import SamplerType
from src.gui.components.dialogs.models.train.classifiers.classifier import ClassifierDialog
from src.gui.widgets import AnimatedToggleSwitch, TunableWidget


class DecisionTreeTrainerDialog(ClassifierDialog):
    """ Training dialog for decision tree classifier.

        Attributes
        ----------
        :param database: (ModelDatabase) active league's model database instance.
        :param league_config: (LeagueConfig) configuration of the league, used to fetch the league's id.
        :param df: (pd.DataFrame) dataframe with the leagued data.
        :param parent: (QWidget | None) optional parent of the dialog.
    """

    def __init__(self, database: ModelDatabase, league_config: LeagueConfig, df: pd.DataFrame, parent: Optional[QWidget] = None):
        self._value_getters = {}
        super().__init__(model_cls=DecisionTree, database=database, league_config=league_config, df=df, parent=parent)

    def _add_hyperparameters(self, frame: QFrame, layout: QVBoxLayout) -> List[TunableWidget]:
        """ Add the decision tree hyperparameter controls. """

        def create_int_slider(
                minimum: int,
                maximum: int,
                step: int,
                default: int,
                tooltip: str,
        ) -> tuple[QWidget, QSlider]:
            control = QWidget(frame)
            control_layout = QHBoxLayout(control)
            control_layout.setContentsMargins(0, 0, 0, 0)
            control_layout.setSpacing(10)

            slider = QSlider(Qt.Orientation.Horizontal, control)
            slider.setRange(minimum, maximum)
            slider.setSingleStep(step)
            slider.setPageStep(step)
            slider.setTickInterval(step)
            slider.setValue(default)
            slider.setToolTip(tooltip)

            value_label = QLabel(str(default), control)
            value_label.setMinimumWidth(28)
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            control_layout.addWidget(slider, 1)
            control_layout.addWidget(value_label)
            control.setFixedWidth(200)
            control.setMinimumHeight(32)

            def update_value(value: int) -> None:
                snapped_value = minimum + round((value - minimum) / step) * step
                snapped_value = min(maximum, max(minimum, snapped_value))
                if snapped_value != value:
                    slider.setValue(snapped_value)
                    return
                value_label.setText(str(value))

            slider.valueChanged.connect(update_value)
            return control, slider

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

        criterion_combo = QComboBox(frame)
        criterion_combo.setFixedWidth(200)
        criterion_combo.setMinimumHeight(32)
        criterion_combo.addItem('Gini', userData='gini')
        criterion_combo.addItem('Log Loss', userData='log_loss')
        criterion_combo.addItem('Entropy', userData='entropy')
        criterion_combo.setToolTip('Objective function to optimize. Gini is usually the best.')

        min_samples_leaf_control, min_samples_leaf_slider = create_int_slider(
            minimum=1,
            maximum=35,
            step=2,
            default=1,
            tooltip='Minimum number of samples required in a leaf node.',
        )
        min_samples_split_control, min_samples_split_slider = create_int_slider(
            minimum=2,
            maximum=30,
            step=2,
            default=2,
            tooltip='Minimum number of samples required to split a node.',
        )

        max_features_combo = QComboBox(frame)
        max_features_combo.setFixedWidth(200)
        max_features_combo.setMinimumHeight(32)
        max_features_combo.addItem('None', userData=None)
        max_features_combo.addItem('Square Root', userData='sqrt')
        max_features_combo.addItem('Log2', userData='log2')
        max_features_combo.setToolTip('Number of features that considered for each split. If None, all features are used.')

        max_depth_control, max_depth_slider = create_int_slider(
            minimum=0,
            maximum=10,
            step=1,
            default=0,
            tooltip='Maximum tree depth. Set 0 to disable depth limit.',
        )

        class_weight_toggle = AnimatedToggleSwitch(checked=True, parent=frame)
        class_weight_toggle.setToolTip('Whether to apply class weights. It is beneficial when imbalanced classes exist.')

        hyperparameter_widgets = [
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
                name='Criterion',
                widget_id='criterion',
                checked=False,
                widget=criterion_combo,
                parent=frame
            ),
            TunableWidget(
                name='Min. Samples Leaf',
                widget_id='min_samples_leaf',
                checked=False,
                widget=min_samples_leaf_control,
                parent=frame
            ),
            TunableWidget(
                name='Min. Samples Split',
                widget_id='min_samples_split',
                checked=False,
                widget=min_samples_split_control,
                parent=frame
            ),
            TunableWidget(
                name='Max. Features',
                widget_id='max_features',
                checked=False,
                widget=max_features_combo,
                parent=frame
            ),
            TunableWidget(
                name='Max. Depth',
                widget_id='max_depth',
                checked=False,
                widget=max_depth_control,
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
            'sampler': sampler_combo.currentData,
            'calibrate_probabilities': calibration_toggle.isChecked,
            'criterion': criterion_combo.currentData,
            'min_samples_leaf': min_samples_leaf_slider.value,
            'min_samples_split': min_samples_split_slider.value,
            'max_features': max_features_combo.currentData,
            'max_depth': max_depth_slider.value,
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
        """ Return the current value of a decision tree hyperparameter. """

        try:
            getter = self._value_getters[name]
        except KeyError as error:
            raise KeyError(f'Unknown decision tree hyperparameter: {name}') from error

        return getter()
