import pandas as pd
from typing import Dict, Any, Type
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QSlider, QVBoxLayout
from superqt import QLabeledSlider
from src.database.model import ModelDatabase
from src.gui.widgets.sliders import add_snap_behavior
from src.gui.windows.models.trainer import TrainerDialog
from src.models.classifiers.decisiontree import DecisionTree


class DecisionTreeTrainerDialog(TrainerDialog):
    """ Decision Tree trainer window. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._criterion_options = {'Gini': 'gini', 'Entropy': 'entropy', 'Log-Loss': 'log_loss'}
        self._leaf_step = 2
        self._samples_step = 2
        self._feature_options = {'None': None, 'SQRT': 'sqrt', 'Log2': 'log2'}
        self._depth_step = 1
        self._class_weights = {'Yes': True, 'No': False}

        self._combo_criterion = None
        self._slider_leaf = None
        self._slider_samples = None
        self._combo_features = None
        self._slider_depth = None
        self._combo_class = None

        super().__init__(
            df=df,
            model_db=model_db,
            title='Decision Tree Trainer',
            width=800,
            height=500,
            supports_calibration=True
        )

    def get_model_cls(self) -> Type:
        return DecisionTree

    def _add_trainer_widgets(self, root: QVBoxLayout):
        row1_box = QHBoxLayout()
        row1_box.setContentsMargins(0, 10, 0, 0)
        row1_box.addStretch(1)

        self._combo_criterion = QComboBox()
        self._combo_criterion.setFixedWidth(90)
        for criterion in self._criterion_options:
            self._combo_criterion.addItem(criterion)
        self._add_tunable_param(
            name='Criterion',
            placeholder_name='criterion',
            widget=self._combo_criterion,
            layout=row1_box,
            tooltip='The objective function of Decision Tree.'
        )

        self._slider_leaf = QLabeledSlider(Qt.Orientation.Horizontal)
        self._slider_leaf.setFixedWidth(150)
        self._slider_leaf.setRange(1, 35)
        self._slider_leaf.setSingleStep(self._leaf_step)
        self._slider_leaf.setTickInterval(4)
        self._slider_leaf.setTickPosition(QSlider.TickPosition.TicksBelow)
        add_snap_behavior(slider=self._slider_leaf, step=self._leaf_step)
        self._add_tunable_param(
            name='Min Samples/Leaf',
            placeholder_name='min_samples_leaf',
            widget=self._slider_leaf,
            layout=row1_box,
            tooltip='Minimum samples to form a leaf (end/target node).'
        )

        self._slider_samples = QLabeledSlider(Qt.Orientation.Horizontal)
        self._slider_samples.setFixedWidth(150)
        self._slider_samples.setRange(2, 30)
        self._slider_samples.setSingleStep(step=self._samples_step)
        self._slider_samples.setTickInterval(4)
        self._slider_samples.setTickPosition(QSlider.TickPosition.TicksBelow)
        add_snap_behavior(slider=self._slider_samples, step=self._samples_step)
        self._add_tunable_param(
            name='Min Samples/Split',
            placeholder_name='min_samples_split',
            widget=self._slider_samples,
            layout=row1_box,
            tooltip='Minimum samples to split a node.'
        )
        row1_box.addStretch(1)
        root.addLayout(row1_box)

        row2_box = QHBoxLayout()
        row2_box.setContentsMargins(0, 10, 0, 0)
        row2_box.addStretch(1)

        self._combo_features = QComboBox()
        self._combo_features.setFixedWidth(80)
        for feature in self._feature_options:
            self._combo_features.addItem(feature)
        self._add_tunable_param(
            name='Max Features',
            placeholder_name='max_features',
            widget=self._combo_features,
            layout=row2_box,
            tooltip='The maximum number of features that is utilized.'
        )

        self._slider_depth = QLabeledSlider(Qt.Orientation.Horizontal)
        self._slider_depth.setFixedWidth(170)
        self._slider_depth.setRange(0, 15)
        self._slider_depth.setSingleStep(self._depth_step)
        self._slider_depth.setTickInterval(1)
        self._slider_depth.setTickPosition(QSlider.TickPosition.TicksBelow)
        add_snap_behavior(slider=self._slider_depth, step=self._depth_step)
        self._add_tunable_param(
            name='Max Depth',
            placeholder_name='max_depth',
            widget=self._slider_depth,
            layout=row2_box,
            tooltip='Maximum depth of tree. Set 0 to allow infinite depth growth.'
        )

        self._combo_class = QComboBox()
        self._combo_class.setFixedWidth(60)
        for class_weight in self._class_weights:
            self._combo_class.addItem(class_weight)
        self._add_tunable_param(
            name='Class Weight',
            placeholder_name='class_weight',
            widget=self._combo_class,
            layout=row2_box,
            tooltip='Whether to balance class weights (weight of each target).'
        )

        row2_box.addStretch(1)
        root.addLayout(row2_box)

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            'criterion': self._criterion_options[self._combo_criterion.currentText()],
            'min_samples_leaf': self._slider_leaf.value(),
            'min_samples_split': self._slider_samples.value(),
            'max_features': self._feature_options[self._combo_features.currentText()],
            'max_depth': self._slider_depth.value(),
            'class_weight': self._class_weights[self._combo_class.currentText()]
        }
