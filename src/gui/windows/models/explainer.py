import pandas as pd
from typing import Type
from abc import abstractmethod
from PyQt6.QtWidgets import QDialog, QCheckBox, QComboBox, QFrame, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QSpinBox, QMessageBox
from src.database.model import ModelDatabase
from src.gui.utils.taskrunner import TaskRunnerDialog
from src.gui.widgets.plot import PlotWindow
from src.interpretability.explainer import ClassifierExplainer
from src.models.model import ClassificationModel
from src.preprocessing.dataset import DatasetPreprocessor, TargetType


class ExplainerDialog(QDialog):
    """ Base class for all trainer dialogs. It utilizes a standard train/eval and tuning procedures for all models. """

    def __init__(
            self,
            df: pd.DataFrame,
            model_db: ModelDatabase,
            title: str,
            width: int,
            height: int
    ):
        super().__init__()

        if width < 250 or height < 550:
            raise ValueError(f'Both width x height should be at least 250 x 600 px, got {width}x{height}.')

        self._df = df
        self._model_db = model_db
        self._title = title
        self._width = width
        self._height = height

        # Adding models of the specified class.
        self._model_ids = []
        model_cls = self._get_model_cls()
        model_ids = model_db.get_model_ids()
        for model_id in model_ids:
            config = model_db.load_model_config(model_id=model_id)

            if config['cls'] == model_cls:
                self._model_ids.append(model_id)

        # Getting input features.
        non_trainable_columns = DatasetPreprocessor().non_trainable_columns
        self._input_features = df.columns.drop(non_trainable_columns).tolist()

        # Initializing available targets.
        self._target_dict = {TargetType.RESULT: ['H', 'D', 'A'], TargetType.OVER_UNDER: ['U', 'O']}

        # Declare placeholders here.
        self._explainer = None
        self._target_type = None

        # Declare UI placeholders here.
        self._combo_model = None
        self._combo_boundary_feat1 = None
        self._combo_boundary_feat2 = None
        self._boundary_btn = None
        self._combo_pdp_feat = None
        self._combo_pdp_targets = None
        self._pdp_btn = None
        self._spin_waterfall_index = None
        self._combo_waterfall_targets = None
        self._waterfall_btn = None
        self._combo_shap_targets = None
        self._check_shap_clusters = None
        self._shap_btn = None

        self._initialize_window()
        self._add_widgets()

    @abstractmethod
    def _get_model_cls(self) -> Type:
        pass

    @abstractmethod
    def _add_additional_widgets(self, root: QVBoxLayout):
        pass

    @abstractmethod
    def _get_explainer(self, model: ClassificationModel) -> ClassifierExplainer:
        pass

    def _initialize_window(self):
        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

    def _add_widgets(self):
        root = QVBoxLayout(self)
        root.addSpacing(15)

        # --- Model selection ---
        model_hbox = QHBoxLayout()
        model_hbox.addStretch(1)

        self._combo_model = QComboBox()
        self._combo_model.setFixedWidth(220)
        for model_id in self._model_ids:
            self._combo_model.addItem(model_id)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.currentIndexChanged.connect(self._on_model_change)
        model_hbox.addWidget(QLabel('Model ID: '))
        model_hbox.addWidget(self._combo_model)
        model_hbox.addStretch(1)
        root.addLayout(model_hbox)

        # Adding Boundary plot.
        boundary_label_row = QHBoxLayout()
        boundary_label_row.setContentsMargins(0, 10, 0, 0)     # Adding 10px top margin.
        boundary_label_row.setSpacing(8)

        # Adding left/right horizontal lines (separators).
        left_line = QFrame()
        left_line.setFrameShape(QFrame.Shape.HLine)
        left_line.setFrameShadow(QFrame.Shadow.Sunken)
        right_line = QFrame()
        right_line.setFrameShape(QFrame.Shape.HLine)
        right_line.setFrameShadow(QFrame.Shadow.Sunken)

        # Stretch to keep the middle centered; lines expand, label stays centered
        boundary_label_row.addStretch(1)
        boundary_label_row.addWidget(left_line, 1)
        boundary_label_row.addWidget(QLabel('Boundary Plot'))
        boundary_label_row.addWidget(right_line, 1)
        boundary_label_row.addStretch(1)
        root.addLayout(boundary_label_row)

        boundary_row = QHBoxLayout()
        boundary_row.addStretch(1)
        self._combo_boundary_feat1 = QComboBox()
        self._combo_boundary_feat1.setFixedWidth(80)
        for feat in self._input_features:
            self._combo_boundary_feat1.addItem(feat)
        self._combo_boundary_feat1.setCurrentIndex(-1)
        boundary_row.addWidget(QLabel('x-Feature: '))
        boundary_row.addWidget(self._combo_boundary_feat1)

        self._combo_boundary_feat2 = QComboBox()
        self._combo_boundary_feat2.setFixedWidth(80)
        for feat in self._input_features:
            self._combo_boundary_feat2.addItem(feat)
        self._combo_boundary_feat2.setCurrentIndex(-1)
        boundary_row.addWidget(QLabel(' y-Feature: '))
        boundary_row.addWidget(self._combo_boundary_feat2)
        boundary_row.addStretch(1)
        root.addLayout(boundary_row)

        boundary_row_btn = QHBoxLayout()
        boundary_row_btn.addStretch(1)
        self._boundary_btn = QPushButton('Plot Boundaries')
        self._boundary_btn.setFixedWidth(150)
        self._boundary_btn.setFixedHeight(30)
        self._boundary_btn.setEnabled(False)
        self._boundary_btn.clicked.connect(self._plot_boundaries)
        boundary_row_btn.addWidget(self._boundary_btn)
        boundary_row_btn.addStretch(1)
        root.addLayout(boundary_row_btn)

        # Adding PDP.
        pdp_label_row = QHBoxLayout()
        pdp_label_row.setContentsMargins(0, 10, 0, 0)     # Adding 10px top margin.
        pdp_label_row.setSpacing(8)

        # Adding left/right horizontal lines (separators).
        left_line = QFrame()
        left_line.setFrameShape(QFrame.Shape.HLine)
        left_line.setFrameShadow(QFrame.Shadow.Sunken)
        right_line = QFrame()
        right_line.setFrameShape(QFrame.Shape.HLine)
        right_line.setFrameShadow(QFrame.Shadow.Sunken)

        # Stretch to keep the middle centered; lines expand, label stays centered
        pdp_label_row.addStretch(1)
        pdp_label_row.addWidget(left_line, 1)
        pdp_label_row.addWidget(QLabel('Partial Dependence Plot'))
        pdp_label_row.addWidget(right_line, 1)
        pdp_label_row.addStretch(1)
        root.addLayout(pdp_label_row)

        pdp_row = QHBoxLayout()
        pdp_row.addStretch(1)
        self._combo_pdp_feat = QComboBox()
        self._combo_pdp_feat.setFixedWidth(80)
        for feat in self._input_features:
            self._combo_pdp_feat.addItem(feat)
        self._combo_pdp_feat.setCurrentIndex(-1)
        pdp_row.addWidget(QLabel('Feature: '))
        pdp_row.addWidget(self._combo_pdp_feat)

        self._combo_pdp_targets = QComboBox()
        self._combo_pdp_targets.setFixedWidth(80)
        pdp_row.addWidget(QLabel(' Target: '))
        pdp_row.addWidget(self._combo_pdp_targets)
        pdp_row.addStretch(1)
        root.addLayout(pdp_row)

        pdp_row_btn = QHBoxLayout()
        pdp_row_btn.addStretch(1)
        self._pdp_btn = QPushButton('Plot Partial Dependence')
        self._pdp_btn.setFixedWidth(220)
        self._pdp_btn.setFixedHeight(30)
        self._pdp_btn.setEnabled(False)
        self._pdp_btn.clicked.connect(self._plot_pdp)
        pdp_row_btn.addWidget(self._pdp_btn)
        pdp_row_btn.addStretch(1)
        root.addLayout(pdp_row_btn)

        # Adding Waterfall Plot.
        waterfall_label_row = QHBoxLayout()
        waterfall_label_row.setContentsMargins(0, 10, 0, 0)     # Adding 10px top margin.
        waterfall_label_row.setSpacing(8)

        # Adding left/right horizontal lines (separators).
        left_line = QFrame()
        left_line.setFrameShape(QFrame.Shape.HLine)
        left_line.setFrameShadow(QFrame.Shadow.Sunken)
        right_line = QFrame()
        right_line.setFrameShape(QFrame.Shape.HLine)
        right_line.setFrameShadow(QFrame.Shadow.Sunken)

        # Stretch to keep the middle centered; lines expand, label stays centered
        waterfall_label_row.addStretch(1)
        waterfall_label_row.addWidget(left_line, 1)
        waterfall_label_row.addWidget(QLabel('Waterfall Plot'))
        waterfall_label_row.addWidget(right_line, 1)
        waterfall_label_row.addStretch(1)
        root.addLayout(waterfall_label_row)

        waterfall_row = QHBoxLayout()
        waterfall_row.addStretch(1)
        self._spin_waterfall_index = QSpinBox()
        self._spin_waterfall_index.setFixedWidth(80)
        self._spin_waterfall_index.setMinimum(0)
        self._spin_waterfall_index.setMaximum(self._df.shape[0])
        self._spin_waterfall_index.setSingleStep(1)
        self._spin_waterfall_index.setValue(0)
        self._spin_waterfall_index.valueChanged.connect(self._set_match_tooltip)
        waterfall_row.addWidget(QLabel('Match Index: '))
        waterfall_row.addWidget(self._spin_waterfall_index)
        self._set_match_tooltip()

        self._combo_waterfall_targets = QComboBox()
        self._combo_waterfall_targets.setFixedWidth(80)
        waterfall_row.addWidget(QLabel(' Target: '))
        waterfall_row.addWidget(self._combo_waterfall_targets)
        waterfall_row.addStretch(1)
        root.addLayout(waterfall_row)

        waterfall_row_btn = QHBoxLayout()
        waterfall_row_btn.addStretch(1)
        self._waterfall_btn = QPushButton('Plot Waterfall')
        self._waterfall_btn.setFixedWidth(220)
        self._waterfall_btn.setFixedHeight(30)
        self._waterfall_btn.setEnabled(False)
        self._waterfall_btn.clicked.connect(self._plot_waterfall)
        waterfall_row_btn.addWidget(self._waterfall_btn)
        waterfall_row_btn.addStretch(1)
        root.addLayout(waterfall_row_btn)

        # Adding Shap Bar Plot.
        shap_label_row = QHBoxLayout()
        shap_label_row.setContentsMargins(0, 10, 0, 0)     # Adding 10px top margin.
        shap_label_row.setSpacing(8)

        # Adding left/right horizontal lines (separators).
        left_line = QFrame()
        left_line.setFrameShape(QFrame.Shape.HLine)
        left_line.setFrameShadow(QFrame.Shadow.Sunken)
        right_line = QFrame()
        right_line.setFrameShape(QFrame.Shape.HLine)
        right_line.setFrameShadow(QFrame.Shadow.Sunken)

        # Stretch to keep the middle centered; lines expand, label stays centered
        shap_label_row.addStretch(1)
        shap_label_row.addWidget(left_line, 1)
        shap_label_row.addWidget(QLabel('Shap Bar Plot'))
        shap_label_row.addWidget(right_line, 1)
        shap_label_row.addStretch(1)
        root.addLayout(shap_label_row)

        shap_row = QHBoxLayout()
        shap_row.addStretch(1)
        self._combo_shap_targets = QComboBox()
        self._combo_shap_targets.setFixedWidth(80)
        shap_row.addWidget(QLabel('Target: '))
        shap_row.addWidget(self._combo_shap_targets)

        self._check_shap_clusters = QCheckBox('Apply Clustering')
        self._check_shap_clusters.setChecked(True)
        self._check_shap_clusters.setFixedWidth(150)
        shap_row.addWidget(self._check_shap_clusters)
        shap_row.addStretch(1)
        root.addLayout(shap_row)

        shap_row_btn = QHBoxLayout()
        shap_row_btn.addStretch(1)
        self._shap_btn = QPushButton('Plot Shap Values')
        self._shap_btn.setFixedWidth(220)
        self._shap_btn.setFixedHeight(30)
        self._shap_btn.setEnabled(False)
        self._shap_btn.clicked.connect(self._plot_shap)
        shap_row_btn.addWidget(self._shap_btn)
        shap_row_btn.addStretch(1)
        root.addLayout(shap_row_btn)

        self._add_additional_widgets(root=root)
        root.addStretch(1)

    def _on_model_change(self):
        model_id = self._combo_model.currentText()
        model, _ = self._model_db.load_model(model_id=model_id)

        # Get explainer.
        self._explainer = self._get_explainer(model=model)

        # Compute Shap Values.
        TaskRunnerDialog(
            title='Shap Computation',
            info='Computing Shap Values. This may take a while...',
            task_fn=self._explainer.compute_shap_values,
            parent=self
        ).run()

        # Enable plots.

        self._boundary_btn.setEnabled(True)
        self._pdp_btn.setEnabled(True)

        if self._explainer.shap_values is None:
            self._waterfall_btn.setEnabled(False)
            self._shap_btn.setEnabled(False)
        else:
            self._waterfall_btn.setEnabled(True)
            self._shap_btn.setEnabled(True)

        # Fill Targets.
        if self._target_type is None or self._target_type != model.target_type:
            self._combo_pdp_targets.clear()
            self._combo_waterfall_targets.clear()
            self._combo_shap_targets.clear()
            for target in self._target_dict[model.target_type]:
                self._combo_pdp_targets.addItem(target)
                self._combo_waterfall_targets.addItem(target)
                self._combo_shap_targets.addItem(target)

    def _set_match_tooltip(self):
        value = self._spin_waterfall_index.value()
        row = self._df.iloc[value]
        tooltip = f'{row["Date"]} - {row["Home"]} vs {row["Away"]} - {row["HG"]}-{row["AG"]}: {row["Result"]}'
        self._spin_waterfall_index.setToolTip(tooltip)

    def _plot_boundaries(self):
        feat1 = self._combo_boundary_feat1.currentText()
        feat2 = self._combo_boundary_feat2.currentText()

        if feat1 == '' or feat2 == '':
            QMessageBox.critical(self, 'Error', 'No features selected.')
            return

        ax = self._explainer.boundary_plot([feat1, feat2])
        PlotWindow(ax=ax, parent=self, title='Boundary Analysis').show()

    def _plot_pdp(self):
        feature = self._combo_pdp_feat.currentText()
        target = self._combo_pdp_targets.currentText()

        if feature == '' or target == '':
            QMessageBox.critical(self, 'Error', 'No features or target selected.')
            return

        ax = self._explainer.partial_dependence_plot(feature=feature, target=target)
        PlotWindow(ax=ax, parent=self, title='Partial Dependence Analysis').show()

    def _plot_waterfall(self):
        match_index = self._spin_waterfall_index.value()
        target = self._combo_waterfall_targets.currentText()

        if target == '':
            QMessageBox.critical(self, 'Error', 'No target selected.')
            return

        ax = self._explainer.instance_waterfall_plot(match_index=match_index, target=target)
        PlotWindow(ax=ax, parent=self, title='Waterfall Analysis').show()

    def _plot_shap(self):
        target = self._combo_shap_targets.currentText()
        clustering = self._check_shap_clusters.isChecked()

        if target == '':
            QMessageBox.critical(self, 'Error', 'No target selected.')
            return

        ax = self._explainer.shap_bar_plot(target=target, clustering=clustering)
        PlotWindow(ax=ax, parent=self, title='Shap Analysis').show()
