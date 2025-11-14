import pandas as pd
from typing import Type
from PyQt6.QtWidgets import QComboBox, QFrame, QLabel, QHBoxLayout, QPushButton, QVBoxLayout, QMessageBox
from src.database.model import ModelDatabase
from src.gui.widgets.plot import PlotWindow
from src.gui.windows.models.explainer import ExplainerDialog
from src.interpretability.explainers.svm import SVMExplainer
from src.models.classifiers.svm import SVM


class SVMExplainerDialog(ExplainerDialog):
    """ Class that supports SVM explanations. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        self._coeff_btn = None
        self._combo_feat1 = None
        self._combo_feat2 = None
        self._visualize_btn = None

        super().__init__(df=df, model_db=model_db, title='Support Vector Machine (SVM) Explainer', width=400, height=620)

    def _get_model_cls(self) -> Type:
        return SVM

    def _add_additional_widgets(self, root: QVBoxLayout):
        # Adding coefficients plot.
        coeff_label_row = QHBoxLayout()
        coeff_label_row.setContentsMargins(0, 10, 0, 0)     # Adding 10px top margin.
        coeff_label_row.setSpacing(8)

        # Adding left/right horizontal lines (separators).
        left_line = QFrame()
        left_line.setFrameShape(QFrame.Shape.HLine)
        left_line.setFrameShadow(QFrame.Shadow.Sunken)
        right_line = QFrame()
        right_line.setFrameShape(QFrame.Shape.HLine)
        right_line.setFrameShadow(QFrame.Shadow.Sunken)

        # Stretch to keep the middle centered; lines expand, label stays centered
        coeff_label_row.addStretch(1)
        coeff_label_row.addWidget(left_line, 1)
        coeff_label_row.addWidget(QLabel('Coefficients Bar Plot'))
        coeff_label_row.addWidget(right_line, 1)
        coeff_label_row.addStretch(1)
        root.addLayout(coeff_label_row)

        coeff_row_btn = QHBoxLayout()
        coeff_row_btn.addStretch(1)
        self._coeff_btn = QPushButton('Plot Coefficients')
        self._coeff_btn.setFixedWidth(150)
        self._coeff_btn.setFixedHeight(30)
        self._coeff_btn.setEnabled(False)
        self._coeff_btn.clicked.connect(self._plot_coeffs)
        coeff_row_btn.addWidget(self._coeff_btn)
        coeff_row_btn.addStretch(1)
        root.addLayout(coeff_row_btn)

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

        features_row = QHBoxLayout()
        features_row.addStretch(1)
        self._combo_feat1 = QComboBox()
        self._combo_feat1.setFixedWidth(80)
        for feat in self._input_features:
            self._combo_feat1.addItem(feat)
        self._combo_feat1.setCurrentIndex(-1)
        features_row.addWidget(QLabel('x-Feature: '))
        features_row.addWidget(self._combo_feat1)

        self._combo_feat2 = QComboBox()
        self._combo_feat2.setFixedWidth(80)
        for feat in self._input_features:
            self._combo_feat2.addItem(feat)
        self._combo_feat2.setCurrentIndex(-1)
        features_row.addWidget(QLabel(' y-Feature: '))
        features_row.addWidget(self._combo_feat2)
        features_row.addStretch(1)
        root.addLayout(features_row)

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

    def _get_explainer(self, model: SVM) -> SVMExplainer:
        return SVMExplainer(model=model, df=self._df)

    def _on_model_change(self):
        super()._on_model_change()

        self._coeff_btn.setEnabled(True)
        self._visualize_btn.setEnabled(True)

        if self._explainer.shap_values is None:
            self._pdp_btn.setEnabled(False)
        else:
            self._pdp_btn.setEnabled(True)

    def _plot_coeffs(self):
        ax = self._explainer.coefficients_bar_plot()
        PlotWindow(ax=ax, parent=self, title='Coefficient Analysis').show()

    def _plot_model(self):
        feat1 = self._combo_feat1.currentText()
        feat2 = self._combo_feat2.currentText()

        if feat1 == '' or feat2 == '':
            QMessageBox.critical(self, 'Error', 'No features selected.')
            return

        ax = self._explainer.visualize_model(features=[feat1, feat2])
        PlotWindow(ax=ax, parent=self, title='Model Visualization').show()
