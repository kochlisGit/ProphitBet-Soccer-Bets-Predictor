import pandas as pd
from typing import Type
from PyQt6.QtWidgets import QFrame, QLabel, QHBoxLayout, QPushButton, QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.widgets.plot import PlotWindow
from src.gui.windows.models.explainer import ExplainerDialog
from src.interpretability.explainers.discriminant import DiscriminantAnalysisExplainer
from src.models.classifiers.discriminant import DiscriminantAnalysisClassifier


class DiscriminantExplainerDialog(ExplainerDialog):
    """ Class that supports LDA/QDA Explainer explanations. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._visualize_btn = None

        super().__init__(df=df, model_db=model_db, title='Discriminant (LDA/QDA) Explainer', width=400, height=550)

    def _get_model_cls(self) -> Type:
        return DiscriminantAnalysisClassifier

    def _add_additional_widgets(self, root: QVBoxLayout):
        # Adding model visualization plot.
        vis_label_row = QHBoxLayout()
        vis_label_row.setContentsMargins(0, 10, 0, 0)     # Adding 10px top margin.
        vis_label_row.setSpacing(8)

        # Adding left/right horizontal lines (separators).
        left_line = QFrame()
        left_line.setFrameShape(QFrame.Shape.HLine)
        left_line.setFrameShadow(QFrame.Shadow.Sunken)
        right_line = QFrame()
        right_line.setFrameShape(QFrame.Shape.HLine)
        right_line.setFrameShadow(QFrame.Shadow.Sunken)

        # Stretch to keep the middle centered; lines expand, label stays centered
        vis_label_row.addStretch(1)
        vis_label_row.addWidget(left_line, 1)
        vis_label_row.addWidget(QLabel('Model Visualization Plot'))
        vis_label_row.addWidget(right_line, 1)
        vis_label_row.addStretch(1)
        root.addLayout(vis_label_row)

        vis_row_btn = QHBoxLayout()
        vis_row_btn.addStretch(1)
        self._visualize_btn = QPushButton('Visualize Model')
        self._visualize_btn.setFixedWidth(150)
        self._visualize_btn.setFixedHeight(30)
        self._visualize_btn.setEnabled(False)
        self._visualize_btn.clicked.connect(self._plot_model)
        vis_row_btn.addWidget(self._visualize_btn)
        vis_row_btn.addStretch(1)
        root.addLayout(vis_row_btn)

    def _get_explainer(self, model: DiscriminantAnalysisClassifier) -> DiscriminantAnalysisExplainer:
        return DiscriminantAnalysisExplainer(model=model, df=self._df)

    def _on_model_change(self):
        super()._on_model_change()

        self._visualize_btn.setEnabled(True)

    def _plot_model(self):
        ax = self._explainer.visualize_model()
        PlotWindow(ax=ax, parent=self, title='Model Visualization').show()
