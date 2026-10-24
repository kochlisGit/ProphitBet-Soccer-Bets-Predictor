from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTabBar, QVBoxLayout, QWidget
from src.core.database.agent import AgentDatabase
from src.core.database.model import ModelDatabase
from src.gui.components.dialogs.tasks import ThreadTask
from src.gui.widgets.listview import ListView
from src.gui.widgets.toggle import AnimatedToggleSwitch


class ManageModels(QDialog):
    """ Review reports for, or delete, stored models and betting agents. """

    def __init__(self, model_database: ModelDatabase, agent_database: AgentDatabase, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._models_tab = 0
        self._agents_tab = 1

        self._league_id = model_database.league_id
        self._model_database = model_database
        self._agent_database = agent_database
        self._model_report_dialog_cls = None
        self._agent_report_dialog_cls = None
        self._report_dialogs = []

        self._tabs = None
        self._report_toggle = None
        self._delete_toggle = None
        self._list_view = None

        self._setup_dialog()
        self._load_resources()
        self._setup_ui()

    @property
    def report_toggle(self) -> AnimatedToggleSwitch:
        return self._report_toggle

    @property
    def delete_toggle(self) -> AnimatedToggleSwitch:
        return self._delete_toggle

    @property
    def list_view(self) -> ListView:
        return self._list_view

    @property
    def tabs(self) -> QTabBar:
        return self._tabs

    def _setup_dialog(self) -> None:
        self.setWindowTitle('Manage Models & Agents')
        self.setWindowIcon(QIcon('storage/icons/menu/models-manage.png'))
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setModal(True)
        self.setMinimumSize(520, 400)
        self.resize(680, 560)

    def _load_resources(self) -> None:
        """Initialize model and agent resources behind a progress dialog."""

        task = ThreadTask(
            self._initialize_resources,
            title='Loading Models',
            description='Loading models',
            parent=self,
        )
        resources = task.run()
        task.deleteLater()

        if resources is None:
            raise RuntimeError('Model and agent resources could not be initialized.')

        self._model_report_dialog_cls, self._agent_report_dialog_cls = resources

    def _initialize_resources(self):
        """Import and initialize model and agent resources in the task thread."""

        from src.gui.components.dialogs.models.train.agents.utils import (
            TrainEvalReportDialog as AgentTrainEvalReportDialog
        )
        from src.gui.components.dialogs.models.train.classifiers.utils import (
            TrainEvalReportDialog as ModelTrainEvalReportDialog
        )

        return ModelTrainEvalReportDialog, AgentTrainEvalReportDialog

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        self._tabs = QTabBar(self)
        self._tabs.addTab('Models')
        self._tabs.addTab('Agents')
        self._tabs.setExpanding(True)
        self._tabs.setDrawBase(True)
        tab_font = self._tabs.font()
        tab_font.setBold(True)
        tab_font.setPointSize(tab_font.pointSize() + 1)
        self._tabs.setFont(tab_font)
        layout.addWidget(self._tabs)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(12)
        mode_layout.addStretch(1)

        report_label = self._create_mode_label('Show Report')
        self._report_toggle = AnimatedToggleSwitch(checked=True, parent=self)
        self._report_toggle.setToolTip('Open the selected item\'s training and evaluation report.')
        mode_layout.addWidget(report_label)
        mode_layout.addWidget(self._report_toggle)

        mode_layout.addSpacing(24)

        delete_label = self._create_mode_label('Delete')
        self._delete_toggle = AnimatedToggleSwitch(checked=False, parent=self)
        self._delete_toggle.setToolTip('Permanently delete the selected item after confirmation.')
        mode_layout.addWidget(delete_label)
        mode_layout.addWidget(self._delete_toggle)
        mode_layout.addStretch(1)
        layout.addLayout(mode_layout)

        self._list_view = ListView(items=self._model_database.get_model_ids(), parent=self)
        layout.addWidget(self._list_view, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        close_button = QPushButton('Close', self)
        close_button.setMinimumSize(105, 36)
        close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        self._tabs.currentChanged.connect(self._refresh_items)
        self._report_toggle.toggled.connect(self._report_mode_changed)
        self._delete_toggle.toggled.connect(self._delete_mode_changed)
        self._list_view.item_pressed.connect(self._item_selected)

    @staticmethod
    def _create_mode_label(text: str) -> QLabel:
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        label.setFont(font)
        return label

    def _report_mode_changed(self, checked: bool) -> None:
        if checked and self._delete_toggle.isChecked():
            self._delete_toggle.setChecked(False)

    def _delete_mode_changed(self, checked: bool) -> None:
        if checked and self._report_toggle.isChecked():
            self._report_toggle.setChecked(False)

    def _refresh_items(self, tab_index: Optional[int] = None) -> None:
        if tab_index is None:
            tab_index = self._tabs.currentIndex()

        if tab_index == self._models_tab:
            item_ids = self._model_database.get_model_ids()
        else:
            item_ids = self._agent_database.get_agent_ids()

        self._list_view.set_items(item_ids)

    def _item_selected(self, item_id: str) -> None:
        if self._report_toggle.isChecked():
            self._show_report(item_id)
        elif self._delete_toggle.isChecked():
            self._confirm_delete(item_id)

    def _show_report(self, item_id: str) -> None:
        if self._tabs.currentIndex() == self._models_tab:
            config = self._model_database.index[self._league_id][item_id]
            dialog = self._model_report_dialog_cls(parent=self, model_config=config)
        else:
            config = self._agent_database.index[self._league_id][item_id]
            dialog = self._agent_report_dialog_cls(parent=self, agent_config=config)

        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda _ = None, report=dialog: self._forget_report_dialog(report))
        self._report_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _forget_report_dialog(self, dialog: QDialog) -> None:
        if dialog in self._report_dialogs:
            self._report_dialogs.remove(dialog)

    def _confirm_delete(self, item_id: str) -> None:
        is_model = self._tabs.currentIndex() == self._models_tab
        entity = 'model' if is_model else 'agent'

        confirmed = QMessageBox.question(
            self,
            f'Delete {entity.title()}?',
            f'Are you sure you want to permanently delete {entity} "{item_id}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

        if not confirmed:
            return

        if is_model:
            self._model_database.delete_model(model_id=item_id)
        else:
            self._agent_database.delete_agent(agent_id=item_id)

        self._refresh_items()

    def done(self, result: int) -> None:
        """ Close child reports before closing the manager dialog. """

        for dialog in tuple(self._report_dialogs):
            dialog.close()
        super().done(result)
