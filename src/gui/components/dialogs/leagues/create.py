import numpy as np
from dataclasses import replace
from typing import Any, Callable, Dict, Optional
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget
)
from src.core.database.league import LeagueDatabase
from src.core.leagues.league import League, LeagueConfig
from src.core.preprocessing.filters import filter_league_odds
from src.core.preprocessing.statistics import StatisticsEngine
from src.gui.components.dialogs.tasks import Task
from src.gui.widgets import FloatDoubleRangeSlider, PrimaryQPushButton


class CreateLeagueDialog(QDialog):
    """ Create League Dialog: It constructs a league configuration, download league data, stores and returns the result.

        Attributes
        ----------
        :param parent: (QWidget | None) dialog's parent widget.
        :param settings: (Dict[str, Any]) settings dictionary.
    """

    league_created = Signal(object, object)

    def __init__(
            self,
            parent: Optional[QWidget],
            settings: Dict[str, Any],
            database: LeagueDatabase,
            current_league_id: Optional[str] = None,
    ):
        super().__init__(parent)

        self._settings = settings if settings is not None else {}
        self._database = database
        self._current_league_id = current_league_id

        self.setWindowTitle('Create League')
        self.setWindowIcon(QIcon('storage/icons/menu/ball.png'))
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumSize(600, 400)
        self.resize(800, 600)
        self.setModal(True)

        dialog_font = self.font()
        dialog_font.setPointSize(dialog_font.pointSize() + 1)
        self.setFont(dialog_font)

        self._min_odd = 1.0
        self._max_odd = 10.0
        self._odd_step = 0.1
        self._stats_tooltips = {
            'HW': 'Recent wins by the home team when playing at home.',
            'HL': 'Recent losses by the home team when playing at home.',
            'HGF': 'Recent goals scored by the home team when playing at home.',
            'HGA': 'Recent goals conceded by the home team when playing at home.',
            'HGD': 'Home goals scored minus home goals conceded (HGF - HGA).',
            'HWGD': 'Recent home-team wins by at least the selected goal margin at home.',
            'HLGD': 'Recent home-team losses by at least the selected goal margin at home.',
            'HW%': 'Home-team win percentage when playing at home.',
            'HL%': 'Home-team loss percentage when playing at home.',
            'AW': 'Recent wins by the away team when playing away.',
            'AL': 'Recent losses by the away team when playing away.',
            'AGF': 'Recent goals scored by the away team when playing away.',
            'AGA': 'Recent goals conceded by the away team when playing away.',
            'AGD': 'Away goals scored minus away goals conceded (AGF - AGA).',
            'AWGD': 'Recent away-team wins by at least the selected goal margin away.',
            'ALGD': 'Recent away-team losses by at least the selected goal margin away.',
            'AW%': 'Away-team win percentage when playing away.',
            'AL%': 'Away-team loss percentage when playing away.',
            'HAGF': 'Difference between home and away recent goals scored (HGF - AGF).',
            'HAGA': 'Difference between home and away recent goals conceded (HGA - AGA).',
            'HAGD': 'Difference between the home and away goal differences (HGD - AGD).',
            'HAWGD': 'Difference between home and away marginal wins (HWGD - AWGD).',
            'HALGD': 'Difference between home and away marginal losses (HLGD - ALGD).',
            'HAGFT': 'Difference in recent total goals scored by both teams (HGFT - AGFT).',
            'HAGAT': 'Difference in recent total goals conceded by both teams (HGAT - AGAT).',
            'HWT': 'Recent wins by the home team across both home and away matches.',
            'HLT': 'Recent losses by the home team across both home and away matches.',
            'HGFT': 'Recent goals scored by the home team across all venues.',
            'HGAT': 'Recent goals conceded by the home team across all venues.',
            'HWT%': 'Home-team win percentage across both home and away matches.',
            'HLT%': 'Home-team loss percentage across both home and away matches.',
            'AWT': 'Recent wins by the away team across both home and away matches.',
            'ALT': 'Recent losses by the away team across both home and away matches.',
            'AGFT': 'Recent goals scored by the away team across all venues.',
            'AGAT': 'Recent goals conceded by the away team across all venues.',
            'AWT%': 'Away-team win percentage across both home and away matches.',
            'ALT%': 'Away-team loss percentage across both home and away matches.',
        }

        # Fetching all available league ids and stat columns.
        self._leagues = self._database.leagues
        self._stored_league_ids = set(self._database.get_league_ids())
        self._stats_columns = StatisticsEngine().stats_columns
        self._checked_columns = self._settings.get('leagues', {}).get('stats_columns')

        # Declaring dataframe and league config placeholders.
        self._downloaded_dataframe = None
        self._filtered_df = None
        self._league_config = None

        # Declaring UI placeholders.
        self._league_combo = None
        self._league_id_edit = None
        self._invalid_id_label = None
        self._year_combo = None
        self._stats_boxes = None
        self._away_stats_boxes = None
        self._store_stats_button = None
        self._window_combo = None
        self._margin_combo = None
        self._odds_1_slider = None
        self._odds_x_slider = None
        self._odds_2_slider = None

        self._build_ui()

    @property
    def result_dataframe(self):
        """Return the dataframe that will be stored when the dialog is accepted."""

        return self._filtered_df

    @property
    def league_config(self) -> Optional[LeagueConfig]:
        """ Return the constructed league configuration. """

        return self._league_config

    def _populate_leagues(self) -> None:
        """ Populate the league selector and select its first entry. """

        previous_block_state = self._league_combo.blockSignals(True)
        self._league_combo.clear()

        for league in self._leagues:
            icon = QIcon(f'storage/icons/leagues/{league.country}.png')
            item = f'{league.name} (from {league.min_year})'
            self._league_combo.addItem(icon, item, league)

        self._league_combo.setCurrentIndex(0)
        self._league_combo.blockSignals(previous_block_state)
        self._league_selected(index=0)

    def _league_selected(self, index: int) -> None:
        """ Populate values that depend on the selected league. """
        
        league = self._league_combo.itemData(index, Qt.ItemDataRole.UserRole)

        if not isinstance(league, League):
            return

        self._league_id_edit.setText(self._find_available_league_id(league))
        self._populate_year_combo(league=league)
        self._reset_download_state()

    def _find_available_league_id(self, league: League) -> str:
        """ Return the first free ``country-name-number`` league identifier. """

        number = 1
        while True:
            league_id = f'{league.country}-{league.name}-{number:02d}'

            if league_id not in self._stored_league_ids:
                return league_id

            number += 1

    def _validate_league_id(self, text: Optional[str] = None) -> bool:
        """ Validate the league id and display the first validation failure. """

        league_id = self._league_id_edit.text() if text is None else text

        if not league_id.strip():
            message = 'League ID is required.'
        elif not league_id[0].isalpha():
            message = 'League ID must start with a letter.'
        elif any(not (character.isalnum() or character in '_-') for character in league_id):
            message = 'Only letters, numbers, "_" and "-" are allowed.'
        elif league_id in self._stored_league_ids:
            message = f'League ID already exists.'
        else:
            self._invalid_id_label.clear()
            self._invalid_id_label.hide()
            return True

        self._invalid_id_label.setText(message)
        self._invalid_id_label.show()
        return False

    def _populate_year_combo(self, league: League) -> None:
        """Populate the valid historical season range for ``league``."""

        self._year_combo.clear()
        self._year_combo.addItems([str(year) for year in range(league.min_year, 2021)])
        self._year_combo.setCurrentIndex(0)

    def _selected_stats_columns(self) -> list[str]:
        """Return the statistic names currently checked by the user."""

        selected_columns = []
        for stats_list in (self._stats_boxes, self._away_stats_boxes):
            for index in range(stats_list.count()):
                item = stats_list.item(index)
                checkbox = stats_list.itemWidget(item)
                if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                    selected_columns.append(checkbox.text())
        return selected_columns

    def _store_statistics(self) -> None:
        """Store the currently checked statistics in the application settings."""

        self._checked_columns = self._selected_stats_columns()
        self._settings.setdefault('leagues', {})['stats_columns'] = list(self._checked_columns)

    def _set_filter_widgets_enabled(self, enabled: bool) -> None:
        """Enable or disable only the interactive widgets in Filters."""

        self._odds_1_slider.setEnabled(enabled)
        self._odds_x_slider.setEnabled(enabled)
        self._odds_2_slider.setEnabled(enabled)
        self._filter_button.setEnabled(enabled)

    def _reset_download_state(self) -> None:
        """Discard data that no longer belongs to the current selection."""

        self._downloaded_dataframe = None
        self._filtered_df = None
        self._league_config = None

        self._download_status_label.setText('No data downloaded')
        self._selected_matches_label.setText('0 / 0 matches filtered')
        self._create_button.setEnabled(False)
        self._set_filter_widgets_enabled(False)

    def _build_league_config(self, stats_columns: list[str]) -> Optional[LeagueConfig]:
        """Build a league configuration from the construction widgets."""

        league = self._league_combo.currentData(Qt.ItemDataRole.UserRole)

        if not isinstance(league, League):
            return None

        return LeagueConfig(
            league=league,
            league_id=self._league_id_edit.text(),
            start_year=int(self._year_combo.currentText()),
            stats_columns=stats_columns,
            match_history_window=int(self._window_combo.currentText()),
            goal_diff_margin=int(self._margin_combo.currentText()),
        )

    def _run_task(self, function: Callable[..., Any], *args: Any, title: str, description: str,) -> Any:
        """Run a callable through the modal task dialog."""

        task = Task(function, *args, title=title, description=description, parent=self)
        return task.run()

    def _download_league(self):
        """ Validate the construction settings and download the selected league. """

        stats_columns = self._selected_stats_columns()
        if not stats_columns:
            QMessageBox.warning(
                self,
                'Statistics Required',
                'Select at least one statistic before downloading data.',
            )
            return

        if not self._validate_league_id():
            return

        league_config = self._build_league_config(stats_columns)
        if league_config is None:
            return

        dataframe = self._run_task(
            self._database.download_league,
            league_config,
            title='Download League',
            description='Downloading and processing league data...',
        )

        if dataframe is None:
            self._reset_download_state()
            return

        self._downloaded_dataframe = dataframe
        self._filtered_df = dataframe
        self._league_config = league_config

        match_count = len(dataframe)
        self._download_status_label.setText(f'{match_count} matches downloaded')
        self._selected_matches_label.setText('0 / 0 matches filtered')
        self._set_filter_widgets_enabled(True)
        self._create_button.setEnabled(True)

        QMessageBox.information(
            self,
            'Download Successful',
            f'Successfully downloaded {match_count} matches.',
        )

    def _get_slider_range(self, slider: FloatDoubleRangeSlider) -> tuple[float, float]:
        """Return a slider range, treating its open upper limit as infinity."""

        lower_value, upper_value = slider.values()

        if upper_value == self._max_odd:
            upper_value = np.inf

        return lower_value, upper_value

    def _selected_odd_ranges(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Return the three currently selected odds ranges."""

        range_1 = self._get_slider_range(slider=self._odds_1_slider)
        range_x = self._get_slider_range(slider=self._odds_x_slider)
        range_2 = self._get_slider_range(slider=self._odds_2_slider)
        return range_1, range_x, range_2

    def _replace_config_filters(
            self,
            odd_1_range: tuple[float, float],
            odd_x_range: tuple[float, float],
            odd_2_range: tuple[float, float],
    ):
        """ Store odds filters on the immutable league configuration. """

        if self._league_config is None:
            return

        self._league_config = replace(
            self._league_config,
            odd_1_mask_range=odd_1_range,
            odd_x_mask_range=odd_x_range,
            odd_2_mask_range=odd_2_range,
        )

    def _filter_matches(self) -> None:
        """Apply the selected odds ranges when the Filter button is clicked."""

        if self._downloaded_dataframe is None:
            self._selected_matches_label.setText('0 / 0 matches filtered')
            return

        odd_1_range, odd_x_range, odd_2_range = self._selected_odd_ranges()
        filtered_df = filter_league_odds(
            df=self._downloaded_dataframe,
            odd_1_mask_range=odd_1_range,
            odd_x_mask_range=odd_x_range,
            odd_2_mask_range=odd_2_range,
        )

        if filtered_df is None:
            self._filtered_df = None
            self._selected_matches_label.setText('0 / 0 matches filtered')
            self._create_button.setEnabled(False)
            QMessageBox.warning(
                self,
                'Filtering Failed',
                'No valid matches that satisfy the selected conditions were found. '
                'Relax the filter constraints and try again.',
            )
            return

        self._filtered_df = filtered_df
        self._replace_config_filters(odd_1_range, odd_x_range, odd_2_range)
        self._create_button.setEnabled(True)
        self._selected_matches_label.setText(
            f'{len(filtered_df)} / {len(self._downloaded_dataframe)} matches are filtered'
        )

    def _create_league(self) -> Optional[tuple[Any, LeagueConfig]]:
        """Persist and return the downloaded league data and configuration."""

        if self._filtered_df is None or self._league_config is None:
            QMessageBox.warning(
                self,
                'League Not Ready',
                'Download valid league data before creating the league.',
            )
            return None

        if not self._validate_league_id():
            return None

        league_id = self._league_id_edit.text()
        if self._league_config.league_id != league_id:
            self._league_config = replace(self._league_config, league_id=league_id)

        display_created_league = True
        if self._current_league_id is not None:
            display_created_league = QMessageBox.question(
                self,
                'Replace Open League?',
                f'League "{self._current_league_id}" is currently open. '
                'Do you want to close it and display the newly created league?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) == QMessageBox.StandardButton.Yes

        self._database.save_league(df=self._filtered_df, league_config=self._league_config)
        self._stored_league_ids.add(self._league_config.league_id)
        result = self._filtered_df, self._league_config

        if display_created_league:
            self.league_created.emit(self._filtered_df, self._league_config)
            self.accept()
        else:
            QMessageBox.information(
                self,
                'League Created',
                f'League "{self._league_config.league_id}" was created successfully.'
            )
            self.reject()

        return result

    def _build_ui(self):
        """Build the Create League dialog interface."""

        style = self.style()

        def create_section_title(text: str) -> QLabel:
            label = QLabel(text)
            font = label.font()
            font.setPointSize(13)
            font.setBold(True)
            label.setFont(font)
            return label

        def create_field_label(text: str) -> QLabel:
            label = QLabel(text)
            font = label.font()
            font.setPointSize(11)
            font.setBold(True)
            label.setFont(font)
            return label

        def create_text_label(text: str = '') -> QLabel:
            label = QLabel(text)
            font = label.font()
            font.setPointSize(11)
            label.setFont(font)
            return label

        def create_card() -> QFrame:
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame.setFrameShadow(QFrame.Shadow.Plain)
            return frame

        # ------------------------------------------------------------------
        # Main scroll area.
        # ------------------------------------------------------------------

        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content_widget = QWidget()

        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        scroll_area.setWidget(content_widget)
        dialog_layout.addWidget(scroll_area)

        # ------------------------------------------------------------------
        # Top area.
        # ------------------------------------------------------------------

        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        # ------------------------------------------------------------------
        # Create / League data.
        # ------------------------------------------------------------------

        league_card = create_card()
        league_card.setMinimumWidth(220)

        league_layout = QVBoxLayout(league_card)
        league_layout.setContentsMargins(12, 12, 12, 12)
        league_layout.setSpacing(8)

        league_layout.addWidget(create_section_title('League'))

        league_description = create_text_label(
            'Select the league and historical data range to download.'
        )
        league_description.setWordWrap(True)
        league_layout.addWidget(league_description)

        league_layout.addSpacing(2)

        league_layout.addWidget(create_field_label('League'))

        self._league_combo = QComboBox()
        self._league_combo.setIconSize(QSize(30, 20))
        self._league_combo.setFixedHeight(40)
        self._league_combo.setMaximumWidth(240)
        self._league_combo.setToolTip('Select the football league whose data will be downloaded.')

        league_layout.addWidget(self._league_combo)

        league_layout.addWidget(create_field_label('League ID*'))

        self._league_id_edit = QLineEdit()
        self._league_id_edit.setFixedHeight(30)
        self._league_id_edit.setMaximumWidth(240)
        self._league_id_edit.setPlaceholderText('e.g. premier_league')
        self._league_id_edit.setClearButtonEnabled(True)
        self._league_id_edit.setToolTip('Enter a unique identifier for the league. The id should start with a letter.')

        league_layout.addWidget(self._league_id_edit)

        self._invalid_id_label = create_text_label()
        self._invalid_id_label.setWordWrap(True)
        self._invalid_id_label.setMaximumWidth(240)
        self._invalid_id_label.setStyleSheet(
            'QLabel { color: #E74C3C; font-weight: 600; }'
        )
        self._invalid_id_label.hide()

        league_layout.addWidget(self._invalid_id_label)

        league_layout.addWidget(create_field_label('Start Year'))

        year_layout = QHBoxLayout()
        year_layout.setSpacing(8)

        self._year_combo = QComboBox()
        self._year_combo.setFixedSize(90, 30)
        self._year_combo.setToolTip('Select the first season to start the download.')

        year_layout.addWidget(self._year_combo)
        year_layout.addStretch()

        league_layout.addLayout(year_layout)

        league_layout.addSpacing(6)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        league_layout.addWidget(separator)

        league_layout.addStretch()
        self._download_button = PrimaryQPushButton(text='Download League Data')
        download_button_font = self._download_button.font()
        download_button_font.setPointSize(download_button_font.pointSize() + 1)
        self._download_button.setFont(download_button_font)
        self._download_button.setFixedHeight(32)
        self._download_button.setMaximumWidth(180)
        self._download_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._download_button.setToolTip('Download league data with the selected configuration.')

        league_layout.addWidget(self._download_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self._download_status_label = create_text_label(text='No data downloaded')
        self._download_status_label.setMaximumWidth(240)

        league_layout.addWidget(self._download_status_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # ------------------------------------------------------------------
        # Statistics.
        # ------------------------------------------------------------------

        stats_card = create_card()
        stats_card.setMinimumWidth(250)

        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.setSpacing(8)

        stats_layout.addWidget(create_section_title('Statistics'))
        stats_description = create_text_label(text='Select the statistics calculated for each match. At least 1 is required.')
        stats_description.setWordWrap(True)

        stats_layout.addWidget(stats_description)

        def create_stats_list() -> QListWidget:
            stats_list = QListWidget()
            stats_list.setMinimumHeight(105)
            stats_list.setMaximumHeight(145)
            stats_list.setMaximumWidth(250)
            stats_list.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            stats_list.setSelectionMode(
                QAbstractItemView.SelectionMode.NoSelection
            )
            return stats_list

        self._stats_boxes = create_stats_list()
        self._away_stats_boxes = create_stats_list()

        for column in self._stats_columns:
            item = QListWidgetItem()
            checkbox = QCheckBox(column)

            checked = self._checked_columns is None or column in self._checked_columns
            checkbox.setChecked(checked)
            tooltip = self._stats_tooltips.get(column)

            if tooltip is not None:
                checkbox.setToolTip(tooltip)

            item.setData(Qt.ItemDataRole.UserRole, checked)

            checkbox.toggled.connect(
                lambda checked, current_item=item:
                current_item.setData(Qt.ItemDataRole.UserRole, checked)
            )

            item.setSizeHint(checkbox.sizeHint())

            if column.startswith('H'):
                stats_list = self._stats_boxes
            elif column.startswith('A'):
                stats_list = self._away_stats_boxes
            else:
                continue

            stats_list.addItem(item)
            stats_list.setItemWidget(item, checkbox)

        stats_lists_layout = QHBoxLayout()
        stats_lists_layout.setSpacing(8)

        home_stats_label = create_field_label('Home Stats')
        home_stats_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        home_stats_layout = QVBoxLayout()
        home_stats_layout.setSpacing(4)
        home_stats_layout.addWidget(home_stats_label)
        home_stats_layout.addWidget(self._stats_boxes)

        away_stats_label = create_field_label('Away Stats')
        away_stats_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        away_stats_layout = QVBoxLayout()
        away_stats_layout.setSpacing(4)
        away_stats_layout.addWidget(away_stats_label)
        away_stats_layout.addWidget(self._away_stats_boxes)

        stats_lists_layout.addLayout(home_stats_layout)
        stats_lists_layout.addLayout(away_stats_layout)
        stats_layout.addLayout(stats_lists_layout)

        stats_controls = QHBoxLayout()
        stats_controls.setSpacing(6)

        select_all_button = QPushButton('Select All')
        select_all_button.setToolTip('Select all stats for home & away teams.')
        select_all_button.setFixedHeight(28)
        clear_stats_button = QPushButton('Clear')
        clear_stats_button.setToolTip('Uncheck all stats for home & away teams.')
        clear_stats_button.setFixedHeight(28)
        self._store_stats_button = QPushButton('Store Statistics')
        self._store_stats_button.setToolTip(
            'Store the checked statistics as the default selection.'
        )
        self._store_stats_button.setFixedHeight(28)

        select_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_stats_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._store_stats_button.setCursor(Qt.CursorShape.PointingHandCursor)

        def set_all_stats(checked: bool):
            for stats_list in (self._stats_boxes, self._away_stats_boxes):
                for index in range(stats_list.count()):
                    item = stats_list.item(index)
                    checkbox = stats_list.itemWidget(item)

                    if checkbox is not None:
                        checkbox.setChecked(checked)

        select_all_button.clicked.connect(lambda: set_all_stats(True))
        clear_stats_button.clicked.connect(lambda: set_all_stats(False))
        self._store_stats_button.clicked.connect(self._store_statistics)

        stats_controls.addWidget(select_all_button)
        stats_controls.addWidget(clear_stats_button)
        stats_controls.addWidget(self._store_stats_button)
        stats_controls.addStretch()

        stats_layout.addLayout(stats_controls)

        history_layout = QHBoxLayout()
        history_layout.setSpacing(8)

        history_label = create_field_label('Match History Window')

        self._window_combo = QComboBox()
        self._window_combo.addItems(['2', '3', '4'])
        self._window_combo.setCurrentText('3')
        self._window_combo.setFixedSize(60, 28)
        self._window_combo.setToolTip('Number of recent matches used for historical statistics.')

        history_layout.addWidget(history_label)
        history_layout.addWidget(self._window_combo)
        history_layout.addStretch()

        stats_layout.addLayout(history_layout)

        margin_layout = QHBoxLayout()
        margin_layout.setSpacing(8)

        margin_label = create_field_label('Goal Difference Margin')

        self._margin_combo = QComboBox()
        self._margin_combo.addItems(['2', '3'])
        self._margin_combo.setCurrentText('2')
        self._margin_combo.setFixedSize(60, 28)
        self._margin_combo.setToolTip(
            'Goal margin used by goal-difference statistics.'
        )

        margin_layout.addWidget(margin_label)
        margin_layout.addWidget(self._margin_combo)
        margin_layout.addStretch()

        stats_layout.addLayout(margin_layout)
        stats_layout.addStretch()

        top_layout.addWidget(league_card, 1)
        top_layout.addWidget(stats_card, 1)

        main_layout.addLayout(top_layout)

        # ------------------------------------------------------------------
        # Match odds filters.
        # ------------------------------------------------------------------

        odds_card = create_card()

        odds_layout = QVBoxLayout(odds_card)
        odds_layout.setContentsMargins(12, 12, 12, 2)
        odds_layout.setSpacing(4)

        odds_layout.addWidget(create_section_title('Filters'))

        odds_description = create_text_label(text='Select the accepted odds range for each match outcome.')
        odds_description.setWordWrap(True)

        odds_layout.addWidget(odds_description)

        filter_widget = QWidget()
        filter_widget.setMaximumWidth(470)

        slider_grid = QGridLayout(filter_widget)
        slider_grid.setContentsMargins(0, 0, 0, 0)
        slider_grid.setHorizontalSpacing(8)
        slider_grid.setVerticalSpacing(8)

        home_label = create_field_label('Home (1)')
        draw_label = create_field_label('Draw (X)')
        away_label = create_field_label('Away (2)')

        self._odds_1_slider = FloatDoubleRangeSlider(
            minimum=self._min_odd,
            maximum=self._max_odd,
            step=self._odd_step,
        )
        self._odds_1_slider.setToolTip('Filters the odds for Home Win (1).')
        self._odds_x_slider = FloatDoubleRangeSlider(
            minimum=self._min_odd,
            maximum=self._max_odd,
            step=self._odd_step,
        )
        self._odds_x_slider.setToolTip('Filters the odds for Draw (X).')
        self._odds_2_slider = FloatDoubleRangeSlider(
            minimum=self._min_odd,
            maximum=self._max_odd,
            step=self._odd_step,
        )
        self._odds_2_slider.setToolTip('Filters the odds for Away Win (2).')

        self._odds_1_slider.setMaximumWidth(370)
        self._odds_x_slider.setMaximumWidth(370)
        self._odds_2_slider.setMaximumWidth(370)

        slider_grid.addWidget(home_label, 0, 0)
        slider_grid.addWidget(self._odds_1_slider, 0, 1)

        slider_grid.addWidget(draw_label, 1, 0)
        slider_grid.addWidget(self._odds_x_slider, 1, 1)

        slider_grid.addWidget(away_label, 2, 0)
        slider_grid.addWidget(self._odds_2_slider, 2, 1)

        slider_grid.setColumnStretch(1, 1)

        filter_actions = QHBoxLayout()
        filter_actions.setSpacing(8)
        filter_actions.addStretch()

        self._filter_button = QPushButton('Filter Matches')
        self._filter_button.setFixedHeight(32)
        self._filter_button.setMaximumWidth(150)
        self._filter_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._filter_button.setToolTip('Apply filters to the downloaded data,')

        filter_actions.addWidget(self._filter_button)
        filter_actions.addStretch()

        slider_grid.addLayout(filter_actions, 3, 1)

        self._selected_matches_label = create_text_label(text='0 / 0 matches filtered')
        self._selected_matches_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_grid.addWidget(self._selected_matches_label, 4, 1)


        filter_container = QHBoxLayout()
        filter_container.addStretch()
        filter_container.addWidget(filter_widget)
        filter_container.addStretch()

        odds_layout.addLayout(filter_container)

        main_layout.addWidget(odds_card)

        # ------------------------------------------------------------------
        # Bottom action bar.
        # ------------------------------------------------------------------

        bottom_separator = QFrame()
        bottom_separator.setFrameShape(QFrame.Shape.HLine)
        bottom_separator.setFrameShadow(QFrame.Shadow.Sunken)

        main_layout.addWidget(bottom_separator)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)
        action_layout.addStretch()

        self._cancel_button = QPushButton('Cancel')
        self._cancel_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton))
        self._cancel_button.setIconSize(QSize(18, 18))
        self._cancel_button.setMinimumSize(105, 36)
        self._cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_button.setToolTip('Cancel the create league operation.')
        self._cancel_button.clicked.connect(self.reject)

        self._create_button = QPushButton('Create League')
        self._create_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self._create_button.setIconSize(QSize(18, 18))
        self._create_button.setMinimumSize(115, 36)
        self._create_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._create_button.setToolTip('Proceed with the create league operation.')
        self._create_button.setDefault(True)
        self._create_button.setEnabled(False)

        action_layout.addWidget(self._cancel_button)
        action_layout.addWidget(self._create_button)

        main_layout.addLayout(action_layout)

        self._league_combo.currentIndexChanged.connect(self._league_selected)
        self._league_id_edit.textChanged.connect(self._validate_league_id)
        self._download_button.clicked.connect(self._download_league)
        self._filter_button.clicked.connect(self._filter_matches)
        self._create_button.clicked.connect(self._create_league)

        self._set_filter_widgets_enabled(False)
        self._populate_leagues()
        self._validate_league_id()
