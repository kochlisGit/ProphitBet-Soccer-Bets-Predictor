from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QWidget
from src.core.database.league import LeagueDatabase
from src.gui.components.workspaces.page import WorkspaceActionCard, WorkspacePage

if TYPE_CHECKING:
    import pandas as pd

    from src.core.leagues.league import LeagueConfig
    from src.gui.components.dialogs.leagues.create import CreateLeagueDialog


class LeaguesWorkspace(WorkspacePage):
    """ Workspace Page for Leagues. It includes 3 options:
            1. Create league.
            2. Load league.
            3. Delete league.

        Attributes
        ----------
        :param settings: Dict[str, Any] application settings dict.
        :param parent: (QWidget | None) parent widget.
    """

    league_created = Signal(object, object)

    def __init__(
            self,
            settings: Dict[str, Any],
            database: LeagueDatabase,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(
            title='Leagues',
            description='Manage your leagues and download historical league data.',
            parent=parent
        )

        self._settings = settings
        self._database = database
        self._active_create_dialog: Optional[CreateLeagueDialog] = None

        # Adding "Create League" option.
        card = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/league-create.png',
            title='Create League',
            description='Select a league and download historical league data, '
                        'configure the desired league statistics, and store them in a database.'
                        ' The league data will be displayed in an Excel-like table.',
            icon_size=125,
        )
        card.setMinimumHeight(175)
        card.clicked.connect(self.open_create_league_dialog)
        self.main_layout.addWidget(card)

        # Adding "Load League" option.
        card = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/league-load.png',
            title='Load League',
            description='Load the data from a previously constructed league.',
            icon_size=125,
        )
        card.setMinimumHeight(175)
        card.clicked.connect(self.open_load_league_dialog)
        self.main_layout.addWidget(card)

        # Adding "Delete League" option.
        card = WorkspaceActionCard(
            icon_filepath='storage/icons/menu/league-delete.png',
            title='Delete Leagues',
            description='Permanently remove stored leagues, historical data, and all related database records '
                        '(models, agents, results, analysis, etc.).',
            icon_size=125,
        )
        card.setMinimumHeight(175)
        card.clicked.connect(self.open_close_league_dialog)
        self.main_layout.addWidget(card)
        self.main_layout.addStretch()

    def open_create_league_dialog(self):
        """ Opens the Create League card's application-modal dialog and returns
            the created dataframe and league config.
        """

        from src.gui.components.dialogs.leagues.create import CreateLeagueDialog

        # Opening dialog.
        current_league_id = getattr(self.window(), 'current_league_id', None)
        dialog = CreateLeagueDialog(
            parent=self.window(),
            settings=self._settings,
            database=self._database,
            current_league_id=current_league_id,
        )
        result = dialog.exec()

        # Fetching dataframe and league config.
        dataframe = dialog.result_dataframe
        league_config = dialog.league_config
        dialog.deleteLater()

        if result == QDialog.DialogCode.Accepted and dataframe is not None and league_config is not None:
            self.league_created.emit(dataframe, league_config)

    def open_load_league_dialog(self):
        """Open the Load League dialog and forward its result to MainMenu."""

        from src.gui.components.dialogs.leagues.load import LoadLeagueDialog

        current_league_id = getattr(self.window(), 'current_league_id', None)
        dialog = LoadLeagueDialog(
            parent=self.window(),
            settings=self._settings,
            database=self._database,
            current_league_id=current_league_id,
        )
        result = dialog.exec()

        dataframe = dialog.result_dataframe
        league_config = dialog.league_config
        dialog.deleteLater()

        if result == QDialog.DialogCode.Accepted and dataframe is not None and league_config is not None:
            self.league_created.emit(dataframe, league_config)

    def open_close_league_dialog(self):
        """ Open the stored-league deletion dialog. """

        from src.gui.components.dialogs.leagues.delete import DeleteLeagueDialog

        current_league_id = getattr(self.window(), 'current_league_id', None)
        dialog = DeleteLeagueDialog(
            parent=self.window(),
            settings=self._settings,
            database=self._database,
            current_league_id=current_league_id,
        )
        dialog.exec()
        dialog.deleteLater()
