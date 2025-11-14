from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDialog, QMessageBox, QPushButton, QVBoxLayout
from src.database.league import LeagueDatabase
from src.gui.utils.taskrunner import TaskRunnerDialog


class DeleteLeagueWindow(QDialog):
    def __init__(self, league_db: LeagueDatabase, current_league_id: Optional[str]):
        super().__init__()

        self._current_league_id = current_league_id
        self._league_db = league_db

        self._league_ids = league_db.get_league_ids()

        self._title = 'Delete League'
        self._width = 350
        self._height = 100

        # UI placeholders
        self._combobox_league = None

        self._initialize_window()
        self._add_widgets()

    def exec(self):
        if len(self._league_ids) == 0:
            QMessageBox.critical(
                self,
                'No Existing Leagues.',
                'There are no existing leagues to delete.',
                QMessageBox.StandardButton.Ok
            )
            return QDialog.Rejected

        super().exec()

    def _initialize_window(self):
        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

    def _add_widgets(self):
        root = QVBoxLayout(self)

        self._combobox_league = QComboBox()
        self._combobox_league.setFixedWidth(250)
        for league_id in self._league_ids:
            self._combobox_league.addItem(league_id)
        root.addWidget(self._combobox_league, alignment=Qt.AlignmentFlag.AlignHCenter)

        download_btn = QPushButton('Delete')
        download_btn.setFixedWidth(160)
        download_btn.setFixedHeight(30)
        download_btn.clicked.connect(self._delete_league)
        root.addWidget(download_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        root.addStretch(1)

    def _delete_league(self):
        """ Deletes the league id and updates the combobox items. """

        cb_index = self._combobox_league.currentIndex()
        league_id = self._league_ids[cb_index]

        if league_id == self._current_league_id:
            QMessageBox.critical(
                self,
                'League is Open',
                f'The league {league_id} is currently open. Close the league before trying to delete it.',
                QMessageBox.StandardButton.Ok
            )
            return

        TaskRunnerDialog(
            title='Delete League',
            info=f'Deleting league: {league_id}...',
            task_fn=lambda: self._league_db.delete_league(league_id=league_id),
            parent=self
        ).run()
        self._league_ids.remove(league_id)
        self._combobox_league.removeItem(cb_index)

        if len(self._league_ids) == 0:
            self.close()
