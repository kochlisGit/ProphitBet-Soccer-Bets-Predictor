import pandas as pd
from typing import Dict, Any, Type
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QSlider, QSpinBox, QDoubleSpinBox, QVBoxLayout
from superqt import QLabeledSlider, QLabeledDoubleSlider
from src.database.model import ModelDatabase
from src.gui.widgets.sliders import add_snap_behavior
from src.gui.windows.models.trainer import TrainerDialog
from src.models.classifiers.neuralnets.nn import NeuralNetwork


class NeuralNetworkTrainerDialog(TrainerDialog):
    """ SVM trainer window. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._activations = {'TanH': 'tanh', 'ReLU': 'relu', 'GELU': 'gelu'}
        self._vsns = {'Yes': True, 'No': False}
        self._layer_normalizations = {'Yes': True, 'No': False}
        self._batch_normalizations = {'No': False, 'Yes': True}
        self._class_weights = {'Yes': True, 'No': False}
        self._optimizers = {'Adam': 'adam', 'AdaBelief': 'adabelief', 'Adan': 'adan', 'Ranger25': 'ranger25'}
        self._lookaheads = {'Yes': True, 'No': False}

        self._hidden_step = 64
        self._dropout_step = 0.1
        self._noise_step = 0.01
        self._smoothing_step = 0.01
        self._learning_rate_step = 0.0005
        self._batch_step = 16
        self._epochs_step = 10
        self._early_stopping_step = 5
        self._decay_patience_step = 5
        self._decay_factor_step = 0.1

        self._slider_hidden_layers = None
        self._spin_units = None
        self._combo_hidden_activation = None
        self._combo_layer_normalization = None
        self._combo_batch_normalization = None
        self._slider_dropout = None
        self._slider_noise = None
        self._combo_class = None
        self._combo_optimizer = None
        self._combo_lookahead = None
        self._slider_smoothing = None
        self._spin_learning_rate = None
        self._spin_batch = None
        self._spin_epochs = None
        self._slider_stopping = None
        self._slider_decay_patience = None
        self._slider_decay_factor = None

        super().__init__(
            df=df,
            model_db=model_db,
            title='Deep Neural Network Trainer',
            width=800,
            height=500,
            supports_calibration=False
        )

    def get_model_cls(self) -> Type:
        return NeuralNetwork

    def _add_trainer_widgets(self, root: QVBoxLayout):
        row1_box = QHBoxLayout()
        row1_box.setContentsMargins(0, 10, 0, 0)
        row1_box.addStretch(1)

        self._slider_hidden_layers = QLabeledSlider(Qt.Orientation.Horizontal)
        self._slider_hidden_layers.setFixedWidth(100)
        self._slider_hidden_layers.setRange(1, 4)
        self._slider_hidden_layers.setSingleStep(1)
        self._slider_hidden_layers.setTickInterval(1)
        self._slider_hidden_layers.setValue(2)
        self._slider_hidden_layers.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._add_tunable_param(
            name='Hidden Layers',
            placeholder_name='hidden_layers',
            widget=self._slider_hidden_layers,
            layout=row1_box,
            tooltip='Number of (hidden) neuron layers.'
        )

        self._spin_units = QSpinBox()
        self._spin_units.setFixedWidth(180)
        self._spin_units.setRange(64, 512)
        self._spin_units.setSingleStep(self._hidden_step)
        self._spin_units.setValue(256)
        self._add_tunable_param(
            name='Hidden Units',
            placeholder_name='hidden_units',
            widget=self._spin_units,
            layout=row1_box,
            tooltip='Number of units per hidden layer.'
        )

        self._combo_hidden_activation = QComboBox()
        self._combo_hidden_activation.setFixedWidth(70)
        for activation in self._activations:
            self._combo_hidden_activation.addItem(activation)
        self._combo_hidden_activation.setCurrentIndex(2)
        self._add_tunable_param(
            name='Activation',
            placeholder_name='hidden_activation',
            widget=self._combo_hidden_activation,
            layout=row1_box,
            tooltip='Hidden activation function of each neuron unit.'
        )

        row1_box.addStretch(1)
        root.addLayout(row1_box)

        row2_box = QHBoxLayout()
        row2_box.setContentsMargins(0, 10, 0, 0)
        row2_box.addStretch(1)

        self._combo_vsn = QComboBox()
        self._combo_vsn.setFixedWidth(70)
        for vsn_option in self._vsns:
            self._combo_vsn.addItem(vsn_option)
        self._add_tunable_param(
            name='Variable Selection Network (VSN)',
            placeholder_name='vsn',
            widget=self._combo_vsn,
            layout=row2_box,
            tooltip='Whether to apply variable selection to inputs.'
        )
        self._combo_vsn.setCurrentIndex(1)

        self._combo_layer_normalization = QComboBox()
        self._combo_layer_normalization.setFixedWidth(70)
        for layer_norm in self._layer_normalizations:
            self._combo_layer_normalization.addItem(layer_norm)
        self._add_tunable_param(
            name='Layer Normalization',
            placeholder_name='layer_normalization',
            widget=self._combo_layer_normalization,
            layout=row2_box,
            tooltip='Whether to apply layer normalization after each layer.'
        )

        self._combo_batch_normalization = QComboBox()
        self._combo_batch_normalization.setFixedWidth(70)
        for batch_norm in self._batch_normalizations:
            self._combo_batch_normalization.addItem(batch_norm)
        self._add_tunable_param(
            name='Batch Normalization',
            placeholder_name='batch_normalization',
            widget=self._combo_batch_normalization,
            layout=row2_box,
            tooltip='Whether to apply batch normalization after each layer.'
        )

        row2_box.addStretch(1)
        root.addLayout(row2_box)

        row3_box = QHBoxLayout()
        row3_box.setContentsMargins(0, 10, 0, 0)
        row3_box.addStretch(1)

        self._slider_dropout = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self._slider_dropout.setFixedWidth(120)
        self._slider_dropout.setRange(0.0, 0.5)
        self._slider_dropout.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_dropout.setTickInterval(0.1)
        self._slider_dropout.setSingleStep(self._dropout_step)
        self._slider_dropout.setValue(0.1)
        add_snap_behavior(slider=self._slider_dropout, step=self._dropout_step)
        self._add_tunable_param(
            name='Dropout Rate',
            placeholder_name='dropout_rate',
            widget=self._slider_dropout,
            layout=row3_box,
            tooltip='Whether to apply dropout activation after each layer.'
        )

        self._slider_noise = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self._slider_noise.setFixedWidth(200)
        self._slider_noise.setRange(0.0, 0.2)
        self._slider_noise.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_noise.setTickInterval(0.02)
        self._slider_noise.setSingleStep(self._noise_step)
        self._slider_noise.setValue(0.1)
        add_snap_behavior(slider=self._slider_noise, step=self._noise_step)
        self._add_tunable_param(
            name='Odd Noise Factor',
            placeholder_name='odd_noise_std',
            widget=self._slider_noise,
            layout=row3_box,
            tooltip='The standard deviation of noise, applied in the odds.'
        )

        self._combo_class = QComboBox()
        self._combo_class.setFixedWidth(70)
        for weight in self._class_weights:
            self._combo_class.addItem(weight)
        self._add_tunable_param(
            name='Class Weight',
            placeholder_name='class_weight',
            widget=self._combo_class,
            layout=row3_box,
            tooltip='Whether to apply class weights. Recommended for imbalanced classes.'
        )

        row3_box.addStretch(1)
        root.addLayout(row3_box)

        row4_box = QHBoxLayout()
        row4_box.setContentsMargins(0, 10, 0, 0)
        row4_box.addStretch(1)

        self._combo_optimizer = QComboBox()
        self._combo_optimizer.setFixedWidth(80)
        for optim in self._optimizers:
            self._combo_optimizer.addItem(optim)
        self._add_tunable_param(
            name='Optimizer',
            placeholder_name='optimizer',
            widget=self._combo_optimizer,
            layout=row4_box,
            tooltip='Training (weight update) optimization algorithm. Adam is recommended.'
        )

        self._combo_lookahead = QComboBox()
        self._combo_lookahead.setFixedWidth(80)
        for weight in self._class_weights:
            self._combo_lookahead.addItem(weight)
        self._add_tunable_param(
            name='Lookahead',
            placeholder_name='lookahead',
            widget=self._combo_lookahead,
            layout=row4_box,
            tooltip='Whether to apply lookahead mechanism to optimizer.'
        )

        self._slider_smoothing = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self._slider_smoothing.setFixedWidth(150)
        self._slider_smoothing.setRange(0.0, 0.1)
        self._slider_smoothing.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_smoothing.setTickInterval(0.01)
        self._slider_smoothing.setSingleStep(self._smoothing_step)
        self._slider_smoothing.setValue(0.1)
        add_snap_behavior(slider=self._slider_smoothing, step=self._smoothing_step)
        self._add_tunable_param(
            name='Label Smoothing Factor',
            placeholder_name='label_smoothing',
            widget=self._slider_smoothing,
            layout=row4_box,
            tooltip='The standard deviation of noise, applied in the odds.'
        )

        row4_box.addStretch(1)
        root.addLayout(row4_box)

        row5_box = QHBoxLayout()
        row5_box.setContentsMargins(0, 10, 0, 0)
        row5_box.addStretch(1)

        self._spin_learning_rate = QDoubleSpinBox()
        self._spin_learning_rate.setFixedWidth(90)
        self._spin_learning_rate.setRange(0.0005, 0.02)
        self._spin_learning_rate.setDecimals(5)
        self._spin_learning_rate.setSingleStep(self._learning_rate_step)
        self._spin_learning_rate.setValue(0.001)
        self._add_tunable_param(
            name='Learning Rate',
            placeholder_name='learning_rate',
            widget=self._spin_learning_rate,
            layout=row5_box,
            tooltip='Optimizer\'s learning rate (used to control the updates).'
        )

        self._spin_batch = QSpinBox(self)
        self._spin_batch.setRange(16, 128)
        self._spin_batch.setValue(16)
        self._spin_batch.setFixedWidth(70)
        self._spin_batch.setSingleStep(self._batch_step)
        self._add_tunable_param(
            name='Batch Size',
            placeholder_name='batch_size',
            widget=self._spin_batch,
            layout=row5_box,
            tooltip='Batch size during training. Large batches speed up training but reduce performance.'
        )

        self._spin_epochs = QSpinBox()
        self._spin_epochs.setFixedWidth(70)
        self._spin_epochs.setRange(10, 100)
        self._spin_epochs.setSingleStep(self._epochs_step)
        self._spin_epochs.setValue(50)
        self._add_tunable_param(
            name='Epochs',
            placeholder_name='epochs',
            widget=self._spin_epochs,
            layout=row5_box,
            tooltip='Number of training iterations.'
        )

        row5_box.addStretch(1)
        root.addLayout(row5_box)

        row6_box = QHBoxLayout()
        row6_box.setContentsMargins(0, 10, 0, 0)
        row6_box.addStretch(1)

        self._slider_stopping = QLabeledSlider(Qt.Orientation.Horizontal)
        self._slider_stopping.setFixedWidth(100)
        self._slider_stopping.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_stopping.setRange(5, 30)
        self._slider_stopping.setSingleStep(self._early_stopping_step)
        self._slider_stopping.setTickInterval(5)
        self._slider_stopping.setValue(15)
        add_snap_behavior(slider=self._slider_stopping, step=self._early_stopping_step)
        self._add_tunable_param(
            name='Early Stopping Epochs',
            placeholder_name='early_stopping_patience',
            widget=self._slider_stopping,
            layout=row6_box,
            tooltip='If performance does not improve after some epochs, it stops training.'
        )

        self._slider_decay_patience = QLabeledSlider(Qt.Orientation.Horizontal)
        self._slider_decay_patience.setFixedWidth(100)
        self._slider_decay_patience.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_decay_patience.setRange(0, 30)
        self._slider_decay_patience.setSingleStep(self._decay_patience_step)
        self._slider_decay_patience.setTickInterval(5)
        self._slider_decay_patience.setValue(10)
        add_snap_behavior(slider=self._slider_decay_patience, step=self._decay_patience_step)
        self._add_tunable_param(
            name='Learning Rate Decay',
            placeholder_name='lr_decay_patience',
            widget=self._slider_decay_patience,
            layout=row6_box,
            tooltip='If performance does not improve after some epochs, it decreases learning rate.'
        )

        self._slider_decay_factor = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self._slider_decay_factor.setFixedWidth(100)
        self._slider_decay_factor.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_decay_factor.setRange(0.2, 0.5)
        self._slider_decay_factor.setSingleStep(self._decay_factor_step)
        self._slider_decay_factor.setTickInterval(0.1)
        add_snap_behavior(slider=self._slider_decay_factor, step=self._decay_factor_step)
        self._add_tunable_param(
            name='Decay Factor',
            placeholder_name='lr_decay_factor',
            widget=self._slider_decay_factor,
            layout=row6_box,
            tooltip='Decay factor of learning rate reduction.'
        )
        row6_box.addStretch(1)
        root.addLayout(row6_box)

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            'hidden_layers': self._slider_hidden_layers.value(),
            'hidden_units': self._spin_units.value(),
            'hidden_activation': self._activations[self._combo_hidden_activation.currentText()],
            'vsn': self._vsns[self._combo_vsn.currentText()],
            'layer_normalization': self._layer_normalizations[self._combo_layer_normalization.currentText()],
            'batch_normalization': self._batch_normalizations[self._combo_batch_normalization.currentText()],
            'dropout_rate': self._slider_dropout.value(),
            'odd_noise_std': self._slider_noise.value(),
            'class_weight': self._class_weights[self._combo_class.currentText()],
            'optimizer': self._optimizers[self._combo_optimizer.currentText()],
            'lookahead': self._lookaheads[self._combo_lookahead.currentText()],
            'label_smoothing': self._slider_smoothing.value(),
            'learning_rate': self._spin_learning_rate.value(),
            'batch_size': self._spin_batch.value(),
            'epochs': self._spin_epochs.value(),
            'early_stopping_patience': self._slider_stopping.value(),
            'lr_decay_patience': self._slider_decay_patience.value(),
            'lr_decay_factor': self._slider_decay_factor.value()
        }

    def _save_trained_model(self, model: NeuralNetwork, model_config: Dict[str, Any]):
        model_config.update({'input_size': model.input_size, 'num_classes': model.num_classes})
        super()._save_trained_model(model=model, model_config=model_config)
