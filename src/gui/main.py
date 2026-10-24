from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent, QIcon, QShowEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLayout, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget
from src.core.database.league import LeagueDatabase
from src.gui.components.sidebar.bar import Sidebar

if TYPE_CHECKING:
    import pandas as pd

    from src.core.database.agent import AgentDatabase
    from src.core.database.model import ModelDatabase
    from src.core.leagues.league import LeagueConfig
    from src.gui.components.dialogs.table import TableDialog


def _initialize_model_database(league_id: str):
    """Import and initialize the active league's model database off the GUI thread."""

    from src.core.database.model import ModelDatabase

    return ModelDatabase(league_id=league_id)


def _initialize_agent_database(league_id: str):
    """Import and initialize the active league's agent database off the GUI thread."""

    from src.core.database.agent import AgentDatabase

    return AgentDatabase(league_id=league_id)


class MainMenu(QWidget):
    """ Main menu class. It constructs the GUI and initializes all components. """

    _WORKSPACE_ATTRIBUTES = (
        '_leagues_workspace',
        '_analysis_workspace',
        '_models_workspace',
        '_predictions_workspace',
        '_settings_workspace',
        '_help_workspace',
    )

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._league_database = LeagueDatabase()
        self._model_database: Optional[ModelDatabase] = None
        self._agent_database: Optional[AgentDatabase] = None
        self._database_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix='DatabaseWarmup',
        )
        self._database_league_id: Optional[str] = None
        self._model_database_future: Optional[Future] = None
        self._agent_database_future: Optional[Future] = None

        # Loading system settings and user preferences.
        with open('storage/configs/settings.json', 'r') as jsonfile:
            self._settings = json.load(jsonfile)

        # Fits comfortably on small laptop.
        window_settings = self._settings['settings']['window']
        self.setMinimumSize(window_settings['min_width'], window_settings['min_height'])

        # Adjusting the preferred GUI size.
        self.resize(window_settings['width'], window_settings['height'])
        self._default_window_title = 'ProphitBet-v3'
        self._default_window_icon = QIcon('storage/icons/menu/ball.png')
        self._reset_window_info()

        # Declaring workspace components.
        self._leagues_workspace = None
        self._analysis_workspace = None
        self._models_workspace = None
        self._predictions_workspace = None
        self._settings_workspace = None
        self._help_workspace = None
        self._league_data_window: Optional[TableDialog] = None
        self._startup_load_scheduled = False
        self._df = None
        self._league_config = None

        # Building UI components.
        self._build_ui()

    @property
    def current_league_id(self) -> Optional[str]:
        """ Return the active league identifier, if a league is open. """

        if self._league_config is None:
            return None
        return self._league_config.league_id

    @property
    def league_database(self) -> LeagueDatabase:
        return self._league_database

    @property
    def model_database(self) -> Optional[ModelDatabase]:
        if self._league_config is None:
            return None
        if self._model_database is None:
            league_id = self._league_config.league_id
            self._start_database_initialization(league_id=league_id)
            future = self._model_database_future
            if future is not None:
                database = future.result()
                if self.current_league_id == league_id and future is self._model_database_future:
                    self._model_database = database
                    self._model_database_future = None
        return self._model_database

    @property
    def agent_database(self) -> Optional[AgentDatabase]:
        if self._league_config is None:
            return None
        if self._agent_database is None:
            league_id = self._league_config.league_id
            self._start_database_initialization(league_id=league_id)
            future = self._agent_database_future
            if future is not None:
                database = future.result()
                if self.current_league_id == league_id and future is self._agent_database_future:
                    self._agent_database = database
                    self._agent_database_future = None
        return self._agent_database

    def showEvent(self, event: QShowEvent) -> None:
        """Load the last league only after the main GUI has been shown."""

        super().showEvent(event)

        if not self._startup_load_scheduled:
            self._startup_load_scheduled = True
            QTimer.singleShot(0, self._load_last_league)


    def closeEvent(self, event: QCloseEvent) -> None:
        """ Called automatically when the window is closed. Stores user's config. """

        try:
            self._close_league_data_window()
            self._shutdown_database_initialization()
            self._settings['settings']['window']['width'] = self.width()
            self._settings['settings']['window']['height'] = self.height()

            with open('storage/configs/settings.json', 'w') as jsonfile:
                json.dump(self._settings, jsonfile, indent=2)

            event.accept()
        except Exception as e:
            print(f'Failed to save settings on exit: {e}')

            event.accept()

    def _start_database_initialization(self, league_id: str) -> None:
        """Warm model and agent databases without blocking the GUI thread."""

        if self.current_league_id != league_id or self._database_executor is None:
            return

        if self._database_league_id != league_id:
            self._discard_database_initialization()
            self._database_league_id = league_id

        if self._model_database is None and self._model_database_future is None:
            self._model_database_future = self._database_executor.submit(
                _initialize_model_database,
                league_id,
            )
        if self._agent_database is None and self._agent_database_future is None:
            self._agent_database_future = self._database_executor.submit(
                _initialize_agent_database,
                league_id,
            )

    def _discard_database_initialization(self) -> None:
        """Forget pending database work when the active league changes."""

        for future in (self._model_database_future, self._agent_database_future):
            if future is not None:
                future.cancel()
        self._model_database_future = None
        self._agent_database_future = None
        self._database_league_id = None
        self._model_database = None
        self._agent_database = None

    def _shutdown_database_initialization(self) -> None:
        """Cancel queued warmups without delaying application shutdown."""

        self._discard_database_initialization()
        if self._database_executor is not None:
            self._database_executor.shutdown(wait=False, cancel_futures=True)
            self._database_executor = None

    @staticmethod
    def _scrollable_page(workspace_page: QWidget) -> QScrollArea:
        """Put a workspace behind a resizable scroll area."""

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setMinimumSize(0, 0)
        scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        workspace_page.setMinimumSize(0, 0)
        workspace_page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        scroll_area.setWidget(workspace_page)
        return scroll_area

    def _create_workspace(self, index: int) -> QWidget:
        """Construct one workspace when it is first requested."""

        if index == 0:
            from src.gui.components.workspaces.pages.leagues import LeaguesWorkspace

            workspace = LeaguesWorkspace(settings=self._settings, database=self._league_database)
            workspace.league_created.connect(self._set_current_league)
            return workspace
        if index == 1:
            from src.gui.components.workspaces.pages.analysis import DataAnalysisWorkspace

            return DataAnalysisWorkspace(settings=self._settings, database=self._league_database)
        if index == 2:
            from src.gui.components.workspaces.pages.models import ModelsWorkspace

            return ModelsWorkspace(settings=self._settings)
        if index == 3:
            from src.gui.components.workspaces.pages.predictions import PredictionsWorkspace

            return PredictionsWorkspace(settings=self._settings)
        if index == 4:
            from src.gui.components.workspaces.pages.settings import SettingsWorkspace

            return SettingsWorkspace(settings=self._settings)
        if index == 5:
            from src.gui.components.workspaces.pages.help import HelpWorkspace

            return HelpWorkspace(settings=self._settings)

        raise IndexError(f'Unknown workspace index: {index}')

    def _ensure_workspace(self, index: int) -> QWidget:
        """Return an existing workspace or build and install it lazily."""

        attribute_name = self._WORKSPACE_ATTRIBUTES[index]
        workspace = getattr(self, attribute_name)
        if workspace is not None:
            return workspace

        workspace = self._create_workspace(index)
        setattr(self, attribute_name, workspace)

        scroll = self._scrollable_page(workspace)
        placeholder = self._workspace_stack.widget(index)
        self._workspace_stack.removeWidget(placeholder)
        self._workspace_stack.insertWidget(index, scroll)
        self._page_scrolls[index] = scroll
        placeholder.deleteLater()
        return workspace

    def _show_workspace(self, index: int) -> None:
        """Build the requested workspace if necessary, then display it."""

        self._ensure_workspace(index)
        self._workspace_stack.setCurrentIndex(index)

    def _build_ui(self):

        root_layout = QHBoxLayout(self)
        root_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(24)

        self._sidebar = Sidebar()
        root_layout.addWidget(self._sidebar)

        workspace_frame = QFrame()
        workspace_frame.setFrameShape(QFrame.Shape.StyledPanel)
        workspace_frame.setMinimumSize(0, 0)
        workspace_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        workspace_layout = QVBoxLayout(workspace_frame)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        self._workspace_stack = QStackedWidget()
        self._workspace_stack.setMinimumSize(0, 0)
        self._workspace_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        workspace_layout.addWidget(self._workspace_stack)
        root_layout.addWidget(workspace_frame, 1)

        # Preserve stable sidebar/stack indexes with lightweight placeholders.
        # Each real page replaces its placeholder on first use.
        self._page_scrolls = [None] * len(self._WORKSPACE_ATTRIBUTES)
        for _ in self._WORKSPACE_ATTRIBUTES:
            self._workspace_stack.addWidget(QWidget())

        self._sidebar.page_requested.connect(self._show_workspace)
        self._show_workspace(0)
        self._set_league_workspaces_enabled(False)

    def _set_league_workspaces_enabled(self, enabled: bool) -> None:
        """Enable workspaces that require an active league."""

        league_workspace_indexes = (1, 2, 3)
        for index in league_workspace_indexes:
            self._sidebar.buttons[index].setEnabled(enabled)

        if not enabled and self._workspace_stack.currentIndex() in league_workspace_indexes:
            self._sidebar.buttons[0].setChecked(True)
            self._show_workspace(0)

    def _load_last_league(self):
        """ Loads last league if this option is set to True. """

        if self._settings['settings']['load_last_league']:
            last_league_id = self._settings['leagues']['last_league']

            if last_league_id is not None:
                if self._league_database.league_exists(league_id=last_league_id):
                    self._set_current_league(
                        df=self._league_database.load_league(league_id=last_league_id),
                        league_config=self._league_database.index[last_league_id],
                    )
                else:
                    self._settings['leagues']['last_league'] = None
        else:
            self._settings['leagues']['last_league'] = None

    def _set_current_league(self, df: Optional[pd.DataFrame] = None, league_config: Optional[LeagueConfig] = None):
        """ Replace the active league after it has been stored successfully. """

        # Checking whether download result is successful.
        if df is not None:
            self._close_current_league(reset_window_info=False)

        # Fetching league dataframe, league config and adding the league id to last league (so it opens on initialize).
        self._df = df
        self._league_config = league_config
        self._settings['leagues']['last_league'] = league_config.league_id

        # Adding info to
        self._sidebar.status.set_state(ready=True, text=league_config.league_id)
        self._set_league_workspaces_enabled(True)
        self.setWindowTitle(league_config.league_id)
        self.setWindowIcon(QIcon(f'storage/icons/leagues/{league_config.league.country}'))

        if df is not None:
            self._open_league_data_window(
                df=df,
                league_id=league_config.league_id,
            )

        league_id = league_config.league_id
        QTimer.singleShot(
            0,
            lambda: self._start_database_initialization(league_id=league_id),
        )

    def _open_league_data_window(self, df: pd.DataFrame, league_id: str) -> None:
        """Build and show the active league's data table."""

        from src.gui.components.dialogs.table import TableDialog

        self._close_league_data_window()
        data_window = TableDialog(
            parent=self,
            title=league_id,
            df=df,
            readonly=True,
            supports_sorting=True,
            supports_query_search=True,
            supports_deletion=False,
            auto_resize=False,
            hide_missing_option=True,
        )

        data_window.destroyed.connect(lambda _=None, window=data_window: self._clear_data_window(window))
        data_window.finished.connect(lambda _result, window=data_window: self._league_data_window_closed(window))
        self._league_data_window = data_window
        data_window.show()
        data_window.raise_()

    def _close_league_data_window(self) -> None:
        if self._league_data_window is None:
            return

        window = self._league_data_window
        self._league_data_window = None
        window.close()

    def _league_data_window_closed(self, window: TableDialog) -> None:
        """Clear the active league when its table is closed by the user."""

        if self._league_data_window is not window:
            return

        self._league_data_window = None
        self._settings['leagues']['last_league'] = None
        self._close_current_league(reset_window_info=True)

    def _clear_data_window(self, window: TableDialog) -> None:
        if self._league_data_window is window:
            self._league_data_window = None

    def _close_current_league(self, reset_window_info: bool = True) -> None:
        """ Clear the active league and synchronize the sidebar status. """

        self._close_league_data_window()
        self._df = None
        self._league_config = None
        self._discard_database_initialization()
        self._set_league_workspaces_enabled(False)

        if reset_window_info:
            self._sidebar.status.set_state(ready=False, text='No League Currently Loaded')
            self._reset_window_info()

    def _reset_window_info(self) -> None:
        """ Restore the application title and default icon. """

        self.setWindowTitle(self._default_window_title)
        self.setWindowIcon(self._default_window_icon)
