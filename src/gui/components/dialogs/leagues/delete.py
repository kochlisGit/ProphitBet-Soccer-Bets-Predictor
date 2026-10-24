from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget
)
from src.core.database.league import LeagueDatabase
from src.gui.widgets.listview import ListView


class DeleteLeagueDialog(QDialog):
    """Delete stored leagues that are not currently open."""

    def __init__(
            self,
            parent: Optional[QWidget],
            settings: Dict[str, Any],
            database: LeagueDatabase,
            current_league_id: Optional[str] = None,
    ):
        super().__init__(parent)

        self._settings = settings if settings is not None else {}
        self._current_league_id = current_league_id
        self._database = database
        self._league_ids = self._database.get_league_ids()

        self.setWindowTitle('Delete Leagues')
        self.setWindowIcon(QIcon('storage/icons/menu/ball.png'))
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumSize(500, 360)
        self.resize(620, 500)
        self.setModal(True)

        dialog_font = self.font()
        dialog_font.setPointSize(dialog_font.pointSize() + 1)
        self.setFont(dialog_font)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        description = QLabel('Select a stored league to delete it permanently.')
        description.setWordWrap(True)
        layout.addWidget(description)

        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(10)

        sort_layout = QHBoxLayout()
        sort_layout.setSpacing(8)

        sort_label = QLabel('Sort leagues by')
        sort_font = self.font()

        if sort_font.pointSize() > 0:
            sort_font.setPointSize(sort_font.pointSize() + 2)
        else:
            sort_font.setPixelSize(sort_font.pixelSize() + 2)

        sort_font.setBold(True)
        sort_label.setFont(sort_font)

        self._sort_combo = QComboBox()
        self._sort_combo.addItems(['Date Created', 'Alphabetically'])
        self._sort_combo.setMinimumWidth(150)
        self._sort_combo.setEnabled(bool(self._league_ids))

        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self._sort_combo)
        sort_layout.addStretch()
        card_layout.addLayout(sort_layout)

        league_icons = {
            league_id: f'storage/icons/leagues/{league_config.league.country}.png'
            for league_id, league_config in self._database.index.items()
        }
        self._list_view = ListView(items=self._league_ids, item_icons=league_icons, parent=self)
        card_layout.addWidget(self._list_view, 1)
        layout.addWidget(card, 1)

        actions = QHBoxLayout()
        actions.addStretch()
        close_button = QPushButton('Close')
        close_button.setMinimumSize(105, 36)
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        self._sort_combo.currentTextChanged.connect(self._sort_leagues)
        self._list_view.item_pressed.connect(self._delete_league)

    def _sort_leagues(self, sort_mode: str) -> None:
        if sort_mode == 'Alphabetically':
            league_ids = sorted(self._league_ids, key=str.casefold)
        else:
            league_ids = list(self._league_ids)

        self._list_view.set_items(league_ids)

    def _delete_league(self, league_id: str) -> None:
        if league_id == self._current_league_id:
            QMessageBox.critical(
                self,
                'League Is Open',
                f'League "{league_id}" is currently open. Close it before deleting it.',
            )
            return

        confirmed = QMessageBox.question(
            self,
            'Delete League?',
            f'Permanently delete league "{league_id}" and all of its stored data?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

        if not confirmed:
            return

        self._database.delete_league(league_id=league_id)
        self._league_ids.remove(league_id)
        self._sort_combo.setEnabled(bool(self._league_ids))
        self._sort_leagues(self._sort_combo.currentText())
