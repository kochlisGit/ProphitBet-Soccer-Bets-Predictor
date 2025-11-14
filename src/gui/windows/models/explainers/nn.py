import pandas as pd
from typing import Type
from PyQt6.QtWidgets import QFrame, QLabel, QHBoxLayout, QPushButton, QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.widgets.plot import PlotWindow
from src.gui.windows.models.explainer import ExplainerDialog
from src.interpretability.explainers.nn import NeuralNetworkExplainer
from src.models.classifiers.neuralnets.nn import NeuralNetwork


class NeuralNetworkExplainerDialog(ExplainerDialog):
    """ Class that supports Neural Network explanations. It's available only if VSN is True. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._importance_btn = None

        super().__init__(df=df, model_db=model_db, title='Neural Network Explainer', width=400, height=600)

    def _get_model_cls(self) -> Type:
        return NeuralNetwork

    def _add_additional_widgets(self, root: QVBoxLayout):
        # Adding coefficients plot.
        importance_label_row = QHBoxLayout()
        importance_label_row.setContentsMargins(0, 10, 0, 0)     # Adding 10px top margin.
        importance_label_row.setSpacing(8)

        # Adding left/right horizontal lines (separators).
        left_line = QFrame()
        left_line.setFrameShape(QFrame.Shape.HLine)
        left_line.setFrameShadow(QFrame.Shadow.Sunken)
        right_line = QFrame()
        right_line.setFrameShape(QFrame.Shape.HLine)
        right_line.setFrameShadow(QFrame.Shadow.Sunken)

        # Stretch to keep the middle centered; lines expand, label stays centered
        importance_label_row.addStretch(1)
        importance_label_row.addWidget(left_line, 1)
        importance_label_row.addWidget(QLabel('Feature Attention Bar Plot'))
        importance_label_row.addWidget(right_line, 1)
        importance_label_row.addStretch(1)
        root.addLayout(importance_label_row)

        importance_row_btn = QHBoxLayout()
        importance_row_btn.addStretch(1)
        self._importance_btn = QPushButton('Plot Feature Attention')
        self._importance_btn.setFixedWidth(180)
        self._importance_btn.setFixedHeight(30)
        self._importance_btn.setEnabled(False)
        self._importance_btn.clicked.connect(self._plot_importance)
        self._importance_btn.setToolTip('Plots the attention per feature. Supported only if VSN is True.')
        importance_row_btn.addWidget(self._importance_btn)
        importance_row_btn.addStretch(1)
        root.addLayout(importance_row_btn)

    def _get_explainer(self, model: NeuralNetwork) -> NeuralNetworkExplainer:
        return NeuralNetworkExplainer(model=model, df=self._df)

    def _on_model_change(self):
        super()._on_model_change()

        if self._explainer.supports_attention:
            self._importance_btn.setEnabled(True)
        else:
            self._importance_btn.setEnabled(False)

    def _plot_importance(self):
        ax = self._explainer.plot_attention_scores()
        PlotWindow(ax=ax, parent=self, title='Feature Attention Analysis').show()
