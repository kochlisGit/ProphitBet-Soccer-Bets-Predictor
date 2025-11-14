from datetime import date
from typing import Optional, Tuple
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QLabel, QLineEdit, QMessageBox,
    QDialog, QFrame, QHBoxLayout, QPushButton, QSpinBox, QVBoxLayout
)
from superqt import QLabeledDoubleRangeSlider
from src.database.league import LeagueDatabase
from src.gui.utils.taskrunner import TaskRunnerDialog
from src.network.leagues.league import League
from src.network.leagues.downloaders.extra import ExtraLeagueDownloader
from src.network.leagues.downloaders.main import MainLeagueDownloader
from src.preprocessing.statistics import StatisticsEngine


class NewLeagueWindow(QDialog):
    """ New League dialog, where user can download and create a new league. """

    def __init__(self, league_db: LeagueDatabase):
        super().__init__()

        self._league_db = league_db

        self._title = 'New League'
        self._width = 800
        self._height = 450
        self._num_rows = 7
        self._start_year_threshold = date.today().year - 4

        # Declare placeholders here.
        self.league_df = None
        self.league = None

        # Fetch all available leagues and the available league columns.
        self._mandatory_columns = {'Date', 'Season', 'Home', 'Away', 'HG', 'AG', 'Result', '1', 'X', '2'}

        # Fetch all statistic columns.
        basic_stats = StatisticsEngine.get_basic_stat_columns()
        extended_stats = StatisticsEngine.get_extended_stat_columns()
        all_stats = basic_stats + extended_stats
        self._all_stats = set(all_stats)
        self._all_columns = MainLeagueDownloader().expected_columns + all_stats
        self._main_columns = set(self._all_columns).difference(self._mandatory_columns)
        self._extra_columns = set(ExtraLeagueDownloader().expected_columns + basic_stats).difference(self._mandatory_columns)
        self._column_tips = {
            'Date': 'Recorded match date',
            'Season': 'Recorded match season',
            'Home': 'Home team name',
            'Away': 'Away team name',
            'HG': 'Goals scored by home team',
            'AG': 'Goals scored by away team',
            'Result': 'Match result: Home (H) / Draw (D) / Away (A)',
            '1': 'Home team wins odds',
            'X': 'Draw odds',
            '2': 'Away team wins odds',
            'HST': 'Shots on target by home team',
            'AST': 'Away on target by away team',
            'HC': 'Home corners',
            'AC': 'Away corners',
            'HW': 'Home last N wins',
            'AW': 'Away last N wins',
            'HL': 'Home last N losses',
            'AL': 'Away last N losses',
            'HGF': 'Home goal forward (sum of goals scored by home team in last N matches)',
            'AGF': 'Away goal forward (sum of goals scored by away team in last N matches)',
            'HAGF': 'Home-Away goal forward difference (HAGF = HGF - AGF)',
            'HGA': 'Home goal against (sum of goals home team received by away teams last N matches)',
            'AGA': 'Away goal against (sum of goals away team received by home teams last N matches)',
            'HAGA': 'Home-Away goal against difference (HAGA = HGA - AGA)',
            'HGD': 'Home goal difference (HDG = HGF - HGA)',
            'AGD': 'Away goal difference (ADG = AGF - AGA)',
            'HAGD': 'Home-Away goal difference difference (HAGD = HGD - AGD)',
            'HWGD': 'Home wins goal with difference margin (sum of last N matches that home team won with difference margin)',
            'AWGD': 'Away wins goal with difference margin (sum of last N matches that away team won with difference margin)',
            'HAWGD': 'Home-Away wins with goal difference margin difference (HAWGD = HWGD - AWGD)',
            'HLGD': 'Home loss goal difference (sum of last N matches that home team lost with difference)',
            'ALGD': 'Away loss goal difference (sum of last N matches that away team lost with difference)',
            'HALGD': 'Home-Away losses with goal difference margin difference (HALGD = HLGD = ALGD)',
            'HW%': 'Home win rate (from the beginning of the season)',
            'HL%': 'Home loss rate (from the beginning of the season)',
            'AW%': 'Away win rate (from the beginning of the season)',
            'AL%': 'Away loss rate (from the beginning of the season)',
            'HSTF': 'Home shots on target forward (sum of shots on targets by home team in last N matches). Requires HST, Main League',
            'ASTF': 'Away shots on target forward (sum of shots on targets by away team in last N matches). Requires AST, Main League',
            'HCF': 'Home corners forward (sum of corners given to home team in last N matches). Requires HCF, Main League',
            'ACF': 'Away corners forward (sum of corners given to away team in last N matches). Requires ACF, Main League'
        }
        self._leagues = self._league_db.leagues

        # UI placeholders
        self._combobox_league = None
        self._line_edit_id = None
        self._odd_sliders = []
        self._start_year_spin = None
        self._match_history_spin = None
        self._goal_diff_spin = None
        self._checkboxes = {}

        self._initialize_window()
        self._add_widgets()

    def _initialize_window(self):
        """ Initializes dialog window. """

        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

    def _add_widgets(self):
        """ Adds dialog widgets. """

        root = QVBoxLayout(self)

        # --- League selection ---
        league_hbox = QHBoxLayout()
        league_hbox.addStretch(1)   # Stretch all to left.
        league_hbox.addWidget(QLabel('Select League:'))

        self._combobox_league = QComboBox()
        self._combobox_league.setFixedWidth(200)
        for i, league in enumerate(self._leagues):
            icon = QIcon(f'storage/graphics/countries/{league.country}.png')
            self._combobox_league.addItem(icon, league.name)
            self._combobox_league.setItemData(i, f'Category: {league.category}', Qt.ItemDataRole.ToolTipRole)
        self._combobox_league.currentIndexChanged.connect(self._set_league_changed)
        league_hbox.addWidget(self._combobox_league)

        self._line_edit_id = QLineEdit()
        self._line_edit_id.setFixedWidth(200)
        self._line_edit_id.setPlaceholderText('Enter a unique league id...')
        league_hbox.addWidget(QLabel('ID: '))
        league_hbox.addWidget(self._line_edit_id)
        league_hbox.addStretch(1)   # Stretch all to right.
        root.addLayout(league_hbox)

        # --- League Filters ---
        row = QHBoxLayout()
        row.setContentsMargins(0, 10, 0, 0)     # Adding 10px top margin.
        row.setSpacing(8)

        # Adding left/right horizontal lines (separators).
        left_line = QFrame()
        left_line.setFrameShape(QFrame.Shape.HLine)
        left_line.setFrameShadow(QFrame.Shadow.Sunken)
        right_line = QFrame()
        right_line.setFrameShape(QFrame.Shape.HLine)
        right_line.setFrameShadow(QFrame.Shadow.Sunken)

        # stretch to keep the middle centered; lines expand, label stays centered
        row.addStretch(1)
        row.addWidget(left_line, 1)
        row.addWidget(QLabel('League Filters'))
        row.addWidget(right_line, 1)
        row.addStretch(1)
        root.addLayout(row)

        # Adding odd range filters.
        filters_hbox = QHBoxLayout()
        filters_hbox.addStretch(1)  # Adding left stretch.

        for odd in ['1', 'X', '2']:
            label = QLabel(f'Odd {odd}: ')
            label.setToolTip(
                f'Include only matches where odd-{odd} is within this range. 10.0 disables the right boundary.'
            )
            label.setStyleSheet('margin-top: 20px;')    # Add margin to align label with the slider.
            filters_hbox.addWidget(label)

            slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
            slider.setRange(1.0, 10.0)
            slider.setSingleStep(0.1)
            slider.setDecimals(1)
            slider.setValue((1.0, 10.0))
            slider.setFixedWidth(200)
            filters_hbox.addWidget(slider)
            self._odd_sliders.append(slider)

        filters_hbox.addStretch(1)  # Adding right stretch to center widgets.
        root.addLayout(filters_hbox)

        # Adding year, match history and goal diff margin filters.
        spinners_hbox = QHBoxLayout()
        spinners_hbox.setContentsMargins(0, 10, 0, 0)     # Adding 10px top margin.
        spinners_hbox.setSpacing(20)
        spinners_hbox.addStretch(1)  # Adding left stretch.

        label = QLabel('Start Year:')
        label.setToolTip(f'Download data range selector: [start_year, {self._start_year_threshold}]')
        spinners_hbox.addWidget(label)
        self._start_year_spin = QSpinBox(self)
        self._start_year_spin.setFixedWidth(100)
        spinners_hbox.addWidget(self._start_year_spin)

        label = QLabel('Match History Window: ')
        label.setToolTip('Number of N previous matches to compute the stats. Typically, it is set to 3 or 4 matches.')
        spinners_hbox.addWidget(label)
        self._match_history_spin = QSpinBox(self)
        self._match_history_spin.setRange(2, 5)
        self._match_history_spin.setValue(3)
        self._match_history_spin.setFixedWidth(100)
        spinners_hbox.addWidget(self._match_history_spin)

        label = QLabel('Goal-Difference Margin: ')
        label.setToolTip('The number of goals that results in early payouts. Typically, it is set to 2 or 3 goals.')
        spinners_hbox.addWidget(label)
        self._goal_diff_spin = QSpinBox(self)
        self._goal_diff_spin.setRange(2, 5)
        self._goal_diff_spin.setValue(2)
        self._goal_diff_spin.setFixedWidth(100)
        spinners_hbox.addWidget(self._goal_diff_spin)

        spinners_hbox.addStretch(1)  # Adding right stretch.
        root.addLayout(spinners_hbox)

        # --- Columns (6 per line) right below the spinners ---
        columns_grid = QGridLayout()
        columns_grid.setContentsMargins(10, 10, 10, 10)
        columns_grid.setSpacing(10)

        self._build_columns_grid(grid=columns_grid)
        root.addLayout(columns_grid)

        # Push all widgets to the top and add a download button on the bottom.
        download_btn = QPushButton('Download')
        download_btn.setFixedWidth(160)
        download_btn.setFixedHeight(30)
        download_btn.clicked.connect(self._download_league)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(download_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # Push all widgets to the top
        root.addStretch(1)

        # Trigger the first selected league.
        self._set_league_changed(index=0)

    def _set_league_changed(self, index: int):
        """ Sets the default league id. """

        league = self._leagues[index]

        def set_default_league_id():
            self._line_edit_id.setText(f'{league.name}-{league.country}-01')

        def set_default_start_year():
            start_year = league.start_year
            self._start_year_spin.setRange(start_year, self._start_year_threshold)
            self._start_year_spin.setValue(start_year)

        def set_available_league_columns():
            category = league.category

            if category == 'main':
                valid_columns = self._main_columns
            elif category == 'extra':
                valid_columns = self._extra_columns
            else:
                raise ValueError(f'Undefined column category: {category}')

            # Check the columns supported by the selected league.
            for col, checkbox in self._checkboxes.items():
                checkbox.setEnabled(col in valid_columns)

        set_default_league_id()
        set_default_start_year()
        set_available_league_columns()

    def _build_columns_grid(self, grid: QGridLayout):
        """ Adds all columns in a grid and builds column dependencies. """

        def link_disable_dependency(key: str, dependent_key: str):
            """ If checkbox is disabled, force dependent checkbox to be disabled+unchecked. """

            key_cb = self._checkboxes[key]
            dependent_cb = self._checkboxes[dependent_key]

            def dependency(checked: bool):
                if not checked:
                    dependent_cb.setChecked(False)
                    dependent_cb.setEnabled(False)
                else:
                    dependent_cb.setEnabled(True)

            key_cb.toggled.connect(dependency)

        # Build all the columns.
        for i, column_name in enumerate(self._all_columns):
            row = i // self._num_rows
            col = i % self._num_rows
            checkbox = QCheckBox(column_name, checked=True, enabled=column_name not in self._mandatory_columns)

            # Adding tooltip to checkbox.
            tip = self._column_tips.get(column_name)
            if tip:
                checkbox.setToolTip(tip)
            grid.addWidget(checkbox, row, col)

            self._checkboxes[column_name] = checkbox

        # Build column dependencies.
        link_disable_dependency(key='HC', dependent_key='HCF')
        link_disable_dependency(key='AC', dependent_key='ACF')
        link_disable_dependency(key='HST', dependent_key='HSTF')
        link_disable_dependency(key='AST', dependent_key='ASTF')

    def _prepare_league_data(self, league: League):
        def get_min_max_odds(slider: QLabeledDoubleRangeSlider) -> Optional[Tuple[float, float]]:
            min_val, max_val = slider.value()

            if min_val == 1.0 and max_val == 10.0:
                return None

            if max_val == 10.0:
                max_val = 1000

            return min_val, max_val

        return league.clone(
            start_year=self._start_year_spin.value(),
            league_id=self._line_edit_id.text(),
            match_history_window=self._match_history_spin.value(),
            goal_diff_margin=self._goal_diff_spin.value(),
            stats_columns=[
                col for col, checkbutton in self._checkboxes.items() if
                checkbutton.isEnabled() and
                checkbutton.isChecked() and
                col in self._all_stats
            ],
            odd_1_range=get_min_max_odds(slider=self._odd_sliders[0]),
            odd_x_range=get_min_max_odds(slider=self._odd_sliders[1]),
            odd_2_range=get_min_max_odds(slider=self._odd_sliders[2])
        )

    def _download_league(self):
        league_id = self._line_edit_id.text()

        if len(league_id) < 1:
            QMessageBox.critical(
                self,
                'League Failed',
                'Failed to create league, as league_id is empty. Please enter a unique league id.'
            )
            return

        league = self._leagues[self._combobox_league.currentIndex()]
        league = self._prepare_league_data(league=league)

        # Validate odds data.
        odd_1_range = league.odd_1_range
        if odd_1_range is not None and league.odd_1_range[1] - league.odd_1_range[0] < 0.5:
            QMessageBox.critical(
                self,
                'Odd Difference',
                f'The max-min odd-1 difference should be at least 0.5.',
                QMessageBox.StandardButton.Ok
            )
            return

        odd_x_range = league.odd_x_range
        if odd_x_range is not None and league.odd_x_range[1] - league.odd_x_range[0] < 0.5:
            QMessageBox.critical(
                self,
                'Odd Difference',
                f'The max-min odd-x difference should be at least 0.5.',
                QMessageBox.StandardButton.Ok
            )
            return

        odd_2_range = league.odd_2_range
        if odd_2_range is not None and league.odd_2_range[1] - league.odd_2_range[0] < 0.5:
            QMessageBox.critical(
                self,
                'Odd Difference',
                f'The max-min odd-2 difference should be at least 0.5.',
                QMessageBox.StandardButton.Ok
            )
            return

        if self._league_db.league_exists(league_id=league_id):
            QMessageBox.critical(
                self,
                'League Exists',
                f'A league already exists with id: {league_id}. Enter another id for this league.',
                QMessageBox.StandardButton.Ok
            )
            return

        # Running download task.
        dialog = TaskRunnerDialog(
            title='League Create',
            info='Initializing league...',
            task_fn=lambda: self._league_db.create_league(league=league),
            parent=self
        )

        # Storing downloaded data (df and id).
        self.league_df = dialog.run()
        self.league = league

        self.close()
