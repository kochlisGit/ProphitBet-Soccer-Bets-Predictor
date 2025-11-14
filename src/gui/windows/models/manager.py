from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDialog, QMessageBox, QPushButton, QHBoxLayout, QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.utils.taskrunner import TaskRunnerDialog
from src.gui.widgets.tables import SimpleTableDialog


class ModelManagerDialogWindow(QDialog):
    """ Manages the trained models (Deletes & Shows stats). """

    def __init__(self, model_db: ModelDatabase):
        super().__init__()

        self._model_db = model_db
        self._model_ids = model_db.get_model_ids()

        self._title = 'Delete Model'
        self._width = 400
        self._height = 120

        # UI placeholders
        self._combobox_model = None

        self._initialize_window()
        self._add_widgets()

    def exec(self):
        if len(self._model_ids) == 0:
            QMessageBox.critical(
                self,
                'No Existing Models.',
                'There are no existing models to delete.',
                QMessageBox.StandardButton.Ok
            )
            return QDialog.Rejected

        super().exec()

    def _initialize_window(self):
        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

    def _add_widgets(self):
        root = QVBoxLayout(self)

        self._combobox_model = QComboBox()
        self._combobox_model.setFixedWidth(250)
        for model_id in self._model_ids:
            self._combobox_model.addItem(model_id)
        root.addWidget(self._combobox_model, alignment=Qt.AlignmentFlag.AlignHCenter)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        metrics_btn = QPushButton('View Metrics')
        metrics_btn.setFixedWidth(160)
        metrics_btn.setFixedHeight(30)
        metrics_btn.clicked.connect(self._view_metrics)
        download_btn = QPushButton('Delete')
        download_btn.setFixedWidth(160)
        download_btn.setFixedHeight(30)
        download_btn.clicked.connect(self._delete_model)

        btn_row.addWidget(metrics_btn)
        btn_row.addWidget(download_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)
        root.addStretch(1)

    def _view_metrics(self):
        """ Opens train/evaluation metrics for the specified model. """

        # Loading model config from database.,
        cb_index = self._combobox_model.currentIndex()
        model_id = self._model_ids[cb_index]
        model_config = self._model_db.load_model_config(model_id=model_id)
        model_results = model_config['train']['results']

        # Viewing model performance metrics.
        if 'tune' in model_results:
            SimpleTableDialog(df=model_results['tune'], parent=self, title='Sliding Cross Validation Results').show()
        if 'cv' in model_results:
            SimpleTableDialog(df=model_results['cv'], parent=self, title='Cross Validation Results').show()
        if 'sliding-cv' in model_results:
            SimpleTableDialog(df=model_results['sliding-cv'], parent=self, title='Sliding Cross Validation Results').show()
        SimpleTableDialog(df=model_results['fit'], parent=self, title='Training Results').show()

    def _delete_model(self):
        """ Deletes the model id and updates the combobox items. """

        cb_index = self._combobox_model.currentIndex()
        model_id = self._model_ids[cb_index]

        TaskRunnerDialog(
            title='Delete Model',
            info=f'Deleting model: {model_id}...',
            task_fn=lambda: self._model_db.delete_model(model_id=model_id),
            parent=self
        ).run()
        self._model_ids.remove(model_id)
        self._combobox_model.removeItem(cb_index)

        if len(self._model_ids) == 0:
            self.close()
