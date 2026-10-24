from typing import Any, Dict, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget
)
from src.core.database.league import LeagueDatabase
from src.core.leagues.league import LeagueConfig
from src.gui.components.dialogs.tasks import Task
from src.gui.widgets import AnimatedToggleSwitch
from src.gui.widgets.listview import ListView


class LoadLeagueDialog(QDialog):
    """Select and load a league stored by :class:`LeagueDatabase`."""

    league_loaded = Signal(object, object)

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
        self._dataframe = None
        self._league_config = None

        self._list_view = None
        self._sort_combo = None
        self._update_toggle = None

        self.setWindowTitle('Load League')
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

    @property
    def result_dataframe(self):
        return self._dataframe

    @property
    def league_config(self) -> Optional[LeagueConfig]:
        return self._league_config

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        update_layout = QHBoxLayout()
        update_layout.setSpacing(12)

        update_label = QLabel('Update')
        update_font = update_label.font()
        update_font.setBold(True)
        update_font.setPointSize(update_font.pointSize() + 2)
        update_label.setFont(update_font)

        self._update_toggle = AnimatedToggleSwitch(checked=True, parent=self)
        self._update_toggle.setToolTip('Download and process the latest league data before loading it.')

        update_layout.addWidget(update_label)
        update_layout.addWidget(self._update_toggle)
        update_layout.addStretch(1)
        layout.addLayout(update_layout)

        description = QLabel('Select a stored league to load its matches and configuration.')
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
        self._sort_combo.setToolTip('Change the order of the stored leagues.')
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
        cancel_button = QPushButton('Cancel')
        cancel_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton))
        cancel_button.setMinimumSize(105, 36)
        cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_button.clicked.connect(self.reject)
        actions.addWidget(cancel_button)
        layout.addLayout(actions)

        self._sort_combo.currentTextChanged.connect(self._sort_leagues)
        self._list_view.item_pressed.connect(self._load_league)

    def _sort_leagues(self, sort_mode: str) -> None:
        if sort_mode == 'Alphabetically':
            league_ids = sorted(self._league_ids, key=str.casefold)
        else:
            league_ids = list(self._league_ids)

        self._list_view.set_items(league_ids)

    def _load_league(self, league_id: str) -> None:
        if league_id == self._current_league_id:
            QMessageBox.information(
                self,
                'League Already Open',
                f'League "{league_id}" is already open.',
            )
            return

        if self._current_league_id is not None:
            replace_current = QMessageBox.question(
                self,
                'Replace Open League?',
                f'League "{self._current_league_id}" is currently open. '
                f'Do you want to close it and load "{league_id}"?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) == QMessageBox.StandardButton.Yes

            if not replace_current:
                return

        league_config = self._database.index.get(league_id)

        if league_config is None:
            QMessageBox.warning(
                self,
                'League Not Found',
                f'The configuration for "{league_id}" could not be found.',
            )
            return

        if self._update_toggle.isChecked():
            task = Task(
                self._database.update_league,
                title='Update League',
                description=f'Updating league "{league_id}". Please wait...',
                parent=self,
                league_id=league_config.league_id,
            )
            df = task.run()
            task.deleteLater()

            if df is None:
                QMessageBox.critical(
                    self,
                    'Update Failed',
                    f'Failed to update league "{league_id}". Loading historical league data.'
                )
        else:
            df = None

        self._dataframe = df if df is not None else self._database.load_league(league_id=league_config.league_id)
        self._league_config = league_config
        self.league_loaded.emit(self._dataframe, league_config)
        self.accept()
