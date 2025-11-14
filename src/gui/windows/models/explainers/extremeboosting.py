import pandas as pd
from typing import Type
from PyQt6.QtWidgets import QFrame, QLabel, QHBoxLayout, QPushButton, QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.widgets.plot import PlotWindow
from src.gui.windows.models.explainer import ExplainerDialog
from src.interpretability.explainers.extremeboosting import ExtremeBoostingExplainer
from src.models.classifiers.extremeboosting import XGBoost


class ExtremeBoostingTreeExplainerDialog(ExplainerDialog):
    """ Class that supports Logistic explanations. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._importance_btn = None

        super().__init__(df=df, model_db=model_db, title='Logistic Regression Explainer', width=400, height=620)

    def _get_model_cls(self) -> Type:
        return XGBoost

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
        importance_label_row.addWidget(QLabel('Importance Bar Plot'))
        importance_label_row.addWidget(right_line, 1)
        importance_label_row.addStretch(1)
        root.addLayout(importance_label_row)

        importance_row_btn = QHBoxLayout()
        importance_row_btn.addStretch(1)
        self._importance_btn = QPushButton('Plot Feature Impurity')
        self._importance_btn.setFixedWidth(180)
        self._importance_btn.setFixedHeight(30)
        self._importance_btn.setEnabled(False)
        self._importance_btn.clicked.connect(self._plot_importance)
        importance_row_btn.addWidget(self._importance_btn)
        importance_row_btn.addStretch(1)
        root.addLayout(importance_row_btn)

    def _get_explainer(self, model: XGBoost) -> ExtremeBoostingExplainer:
        return ExtremeBoostingExplainer(model=model, df=self._df)

    def _on_model_change(self):
        super()._on_model_change()

        self._importance_btn.setEnabled(True)

    def _plot_importance(self):
        ax = self._explainer.feature_impurity_bar_plot()
        PlotWindow(ax=ax, parent=self, title='Feature Impurity Analysis').show()
