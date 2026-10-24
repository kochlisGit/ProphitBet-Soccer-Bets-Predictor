from typing import Any, Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QStyle, QVBoxLayout, QWidget
from src.core.database.model import ModelDatabase
from src.core.models import classifiers
from src.gui.components.dialogs.models.interpretability import explainers
from src.gui.widgets.listview import ListView


class ExplainerMenuDialog(QDialog):
    """Select a stored model to interpret.

    The model index order is its creation-date order.  Model-specific
    explainer dialogs will be registered in ``_explainer_dialogs`` as they are
    implemented; until then selecting a model simply closes this selector.
    """

    def __init__(self, model_database: ModelDatabase, df: Optional[Any] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._explainers = {
            classifiers.LogisticRegressor: explainers.LogisticRegressionExplainerDialog,
            classifiers.DiscriminantAnalysis: explainers.DiscriminantAnalysisExplainerDialog,
            classifiers.SupportVectorMachine: explainers.SupportVectorMachineExplainerDialog,
            classifiers.KNearestNeighbors: explainers.PermutationExplainerDialog,
            classifiers.NaiveBayes: explainers.PermutationExplainerDialog,
            classifiers.DecisionTree: explainers.DecisionTreeExplainerDialog,
            classifiers.RandomForest: explainers.EnsembleRegressionExplainerDialog,
            classifiers.XGBoost: explainers.EnsembleRegressionExplainerDialog,
            classifiers.LightGBM: explainers.EnsembleRegressionExplainerDialog
        }
        self.SORT_BY_DATE = 'Sort by Date'
        self.SORT_BY_NAME = 'Sort by Name'

        self._model_database = model_database
        self._df = df
        self._model_index = self._model_database.index.get(self._model_database.league_id, {})
        self._model_ids = list(self._model_index)
        self._selected_model_id = None
        self._selected_model_type = None

        self._sort_combo = None
        self._sort_by_type_checkbox = None
        self._list_view = None

        self.setWindowTitle('Explainer Dialog')
        self.setWindowIcon(QIcon('storage/icons/menu/models-explainability.png'))
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumSize(500, 360)
        self.resize(620, 500)
        self.setModal(True)

        dialog_font = self.font()
        dialog_font.setPointSize(dialog_font.pointSize() + 1)
        self.setFont(dialog_font)

        self._build_ui()

    @property
    def selected_model_id(self) -> Optional[str]:
        return self._selected_model_id

    @property
    def selected_model_type(self):
        return self._selected_model_type

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        description = QLabel('Select a trained model to interpret and explain.')
        description.setWordWrap(True)
        layout.addWidget(description)

        card = QFrame(self)
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(10)

        sort_layout = QHBoxLayout()
        sort_layout.setSpacing(8)

        sort_label = QLabel('Sort models by', card)
        sort_font = self.font()
        if sort_font.pointSize() > 0:
            sort_font.setPointSize(sort_font.pointSize() + 2)
        else:
            sort_font.setPixelSize(sort_font.pixelSize() + 2)
        sort_font.setBold(True)
        sort_label.setFont(sort_font)

        self._sort_combo = QComboBox(card)
        self._sort_combo.addItems([self.SORT_BY_DATE, self.SORT_BY_NAME])
        self._sort_combo.setMinimumWidth(150)
        self._sort_combo.setToolTip('Choose the order of the stored models.')
        self._sort_combo.setEnabled(bool(self._model_ids))

        self._sort_by_type_checkbox = QCheckBox('Sort by Type', card)
        self._sort_by_type_checkbox.setToolTip('Group models by classifier type before applying the selected sort order.')
        self._sort_by_type_checkbox.setEnabled(bool(self._model_ids))

        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self._sort_combo)
        sort_layout.addSpacing(12)
        sort_layout.addWidget(self._sort_by_type_checkbox)
        sort_layout.addStretch(1)
        card_layout.addLayout(sort_layout)

        self._list_view = ListView(items=self._model_ids, parent=card)
        card_layout.addWidget(self._list_view, 1)
        layout.addWidget(card, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel_button = QPushButton('Cancel', self)
        cancel_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton))
        cancel_button.setMinimumSize(105, 36)
        cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_button.clicked.connect(self.reject)
        actions.addWidget(cancel_button)
        layout.addLayout(actions)

        self._sort_combo.currentTextChanged.connect(self._sort_models)
        self._sort_by_type_checkbox.toggled.connect(self._sort_models)
        self._list_view.item_pressed.connect(self._model_selected)

    @staticmethod
    def _model_type(model_config):
        """Return the classifier class stored in a model configuration."""

        return model_config.classifier_config.get('model_cls')

    @classmethod
    def _model_type_name(cls, model_config) -> str:
        model_type = cls._model_type(model_config)
        if isinstance(model_type, str):
            return model_type.rsplit(':', maxsplit=1)[-1]
        return getattr(model_type, '__name__', str(model_type))

    def _sort_models(self, _value=None) -> None:
        """Sort model ids using type as an optional primary key."""

        model_ids = list(self._model_ids)
        sort_by_name = self._sort_combo.currentText() == self.SORT_BY_NAME

        if self._sort_by_type_checkbox.isChecked():
            # Python's stable sort preserves model-index order within a type
            # when date is selected.
            if sort_by_name:
                model_ids.sort(
                    key=lambda model_id: (
                        self._model_type_name(self._model_index[model_id]).casefold(),
                        model_id.casefold(),
                    )
                )
            else:
                model_ids.sort(
                    key=lambda model_id: self._model_type_name(
                        self._model_index[model_id]
                    ).casefold()
                )
        elif sort_by_name:
            model_ids.sort(key=str.casefold)

        self._list_view.set_items(model_ids)

    def _model_selected(self, model_id: str) -> None:
        """Resolve the model type and dispatch to its explainer dialog."""

        model_config = self._model_index.get(model_id)

        if model_config is None:
            return

        self._selected_model_id = model_id
        self._selected_model_type = model_config.classifier_config['model_cls']

        explainer_dialog_cls = self._explainers.get(self._selected_model_type, None)
        if explainer_dialog_cls is None and self._model_type_name(model_config) == 'NeuralNetwork':
            from src.gui.components.dialogs.models.interpretability.explainers.neuralnets import (
                NeuralNetsExplainerDialog,
            )

            explainer_dialog_cls = NeuralNetsExplainerDialog
        self.accept()

        if explainer_dialog_cls is None:
            return

        dialog = explainer_dialog_cls(
            model_database=self._model_database,
            model_id=model_id,
            df=self._df,
            parent=self.parentWidget()
        )
        dialog.exec()
        dialog.deleteLater()
