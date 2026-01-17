import ast
import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional
from PyQt6.QtCore import QDate, QTimer
from PyQt6.QtWidgets import QDialog, QLabel, QComboBox, QDateEdit, QFileDialog, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout
from src.database.model import ModelDatabase
from src.gui.widgets.comboboxes import CheckableComboBox
from src.gui.widgets.tables import ExcelTable, StylizedTable
from src.network.fixtures.footystats.scraper import FootyStatsScraper
from src.network.fixtures.utils import match_fixture_teams
from src.network.leagues.league import League
from src.preprocessing.utils.inputs import construct_inputs_by_fixture
from src.preprocessing.utils.target import TargetType


class FixturesDialog(QDialog):
    """ Fixtures dialog which downloads the upcoming league's fixture and makes predictions. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase, league: League):
        super().__init__()

        self._df = df.reset_index(drop=True)
        self._model_db = model_db
        self._league = league

        self._model_ids = model_db.get_model_ids()
        self._title = 'Fixtures Dialog'
        self._width = 800
        self._height = 450

        # Declare placeholders.
        self._y_prob = None
        self._y_pred = None
        self._percentiles = None
        self._odds = None
        self._odd_mask = None
        self._index = None

        self._target_types = {'Result (1/X/2)': TargetType.RESULT, 'U/O-2.5': TargetType.OVER_UNDER}
        self._home_teams = sorted(df['Home'].unique().tolist())
        self._away_teams = sorted(df['Away'].unique().tolist())
        self._result_model_ids = []
        self._uo_model_ids = []
        for model_id in self._model_ids:
            config = model_db.load_model_config(model_id=model_id)
            if config['target_type'] == TargetType.RESULT:
                self._result_model_ids.append(model_id)
            else:
                self._uo_model_ids.append(model_id)

        # Declare UI Placeholders.
        self._calendar = None
        self._combo_model = None
        self._combo_target = None
        self._export_btn = None
        self._combo_filters = None
        self._table = None

        self._initialize_window()
        self._add_widgets()

    def exec(self):
        if len(self._model_ids) == 0:
            QMessageBox.critical(
                self,
                'No Existing Models.',
                'There are no existing models to predict fixtures.',
                QMessageBox.StandardButton.Ok
            )
            return QDialog.Rejected

        super().exec()

    def _initialize_window(self):
        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

    def _add_widgets(self):
        root = QVBoxLayout(self)

        # --- Date/Model initialization ---
        model_hbox = QHBoxLayout()
        model_hbox.addStretch(1)

        td = timedelta(days=180)
        today = date.today()
        q_today = QDate(today.year, today.month, today.day)
        q_min = QDate((today - td).year, (today - td).month, (today - td).day)
        q_max = QDate((today + td).year, (today + td).month, (today + td).day)
        self._calendar = QDateEdit(self)
        self._calendar.setCalendarPopup(True)
        self._calendar.setDate(q_today)
        self._calendar.setDateRange(q_min, q_max)
        self._calendar.setDisplayFormat('yyyy-MM-dd')
        self._calendar.dateChanged.connect(lambda qdate: QTimer.singleShot(50, lambda: self._on_date_change(qdate)))
        model_hbox.addWidget(QLabel('Fixture Date: '))
        model_hbox.addWidget(self._calendar)

        self._combo_target = QComboBox()
        self._combo_target.setFixedWidth(120)
        for target in self._target_types:
            self._combo_target.addItem(target)
        self._combo_target.setCurrentIndex(-1)
        self._combo_target.setEnabled(False)
        self._combo_target.currentIndexChanged.connect(self._on_target_change)
        model_hbox.addWidget(QLabel(' Target: '))
        model_hbox.addWidget(self._combo_target)

        self._combo_model = QComboBox()
        self._combo_model.setFixedWidth(220)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.setEnabled(False)
        self._combo_model.currentIndexChanged.connect(self._on_model_change)
        model_hbox.addWidget(QLabel(' Model ID: '))
        model_hbox.addWidget(self._combo_model)
        model_hbox.addStretch(1)
        root.addLayout(model_hbox)

        filters_hbox = QHBoxLayout()
        filters_hbox.addStretch(1)
        self._combo_filters = CheckableComboBox()
        self._combo_filters.setFixedWidth(180)
        self._combo_filters.setEnabled(False)
        self._combo_filters.checkedItemsChanged.connect(self._on_filters_change)
        filters_hbox.addWidget(QLabel('Filters: '))
        filters_hbox.addWidget(self._combo_filters)
        filters_hbox.addStretch(1)
        root.addLayout(filters_hbox)

        export_hbox = QHBoxLayout()
        export_hbox.addStretch(1)
        self._export_btn = QPushButton('Export')
        self._export_btn.setFixedWidth(100)
        self._export_btn.setFixedHeight(30)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        export_hbox.addWidget(self._export_btn)
        export_hbox.addStretch(1)
        root.addLayout(export_hbox)

        empty_row = ['']*10
        table_df = pd.DataFrame({
            'Home': empty_row,
            'Away': empty_row,
            '1': empty_row,
            'X': empty_row,
            '2': empty_row,
            'Predicted': empty_row,
            'Prob(1)': empty_row,
            'Prob(X)': empty_row,
            'Prob(2)': empty_row,
            'Prob(U)': empty_row,
            'Prob(O)': empty_row
        })
        self._table = ExcelTable(
            parent=self,
            df=table_df,
            readonly=False,
            supports_sorting=False,
            supports_query_search=True,
            supports_deletion=True
        )
        self._table = StylizedTable().stylize_table(table=self._table, options_dict={0: self._home_teams, 1: self._away_teams})
        self._table.setColumnWidth(0, 150)
        self._table.setColumnWidth(1, 150)
        self._table.hide_columns(columns=['Prob(1)', 'Prob(X)', 'Prob(2)', 'Prob(U)', 'Prob(O)'], hide=True)
        root.addWidget(self._table)

    def _on_date_change(self, qdate: QDate):
        # Fetching date.
        date_str = qdate.toPyDate().strftime('%b %d').replace(' 0', ' ')

        # Scraping fixtures.
        scraper = FootyStatsScraper()
        scraper.load_page(self._league.fixture)
        df = scraper.parse_fixture_table(date_str=date_str)

        if df is None:
            QMessageBox.critical(self, 'Parsing Failed', 'Failed to parse fixtures. Make sure the selected date is correct.')
            scraper.quit()
            return

        scraper.quit()

        # Matching fixtures.
        fixtures_df = match_fixture_teams(parsed_teams_df=df, league_df=self._df)

        # Validate matches.
        valid_home_mask = fixtures_df['Home'].isin(self._home_teams)
        valid_away_mask = fixtures_df['Away'].isin(self._away_teams)
        valid_mask = valid_home_mask & valid_away_mask
        rows_dropped = valid_mask.sum() != fixtures_df.shape[0]
        fixtures_df = fixtures_df[valid_mask].reset_index(drop=True)

        # Add self._fixtures_df to table.
        columns = ['Home', 'Away', '1', 'X', '2']
        self._table.clearContents()
        self._table.modify_columns(columns=columns, data=fixtures_df[columns].to_numpy().tolist())

        # Erase History.
        self._y_prob = None
        self._y_pred = None
        self._percentiles = None
        self._odds = None
        self._odd_mask = None

        # Enable target.
        self._combo_target.setEnabled(True)
        self._combo_target.blockSignals(True)
        self._combo_target.setCurrentIndex(-1)
        self._combo_target.blockSignals(False)

        # Clear & Disable models and filters.
        self._combo_model.blockSignals(True)
        self._combo_model.clear()
        self._combo_model.setEnabled(False)
        self._combo_model.blockSignals(False)
        self._combo_filters.blockSignals(True)
        self._combo_filters.clear()
        self._combo_filters.setEnabled(False)
        self._combo_filters.blockSignals(False)

        # Disable export button.
        self._export_btn.setEnabled(False)

        if rows_dropped:
            QMessageBox.information(self, 'Insufficient Data', 'Some matches have been dropped due to insufficient historical data.')

    def _on_target_change(self):
        """ Adds model ids based on the selected target. """

        # Erasing history.
        self._y_prob = None
        self._y_pred = None
        self._percentiles = None

        empty_cols = ['']*10
        self._table.modify_columns(
            columns=['Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)', 'Prob(U)', 'Prob(O)'],
            data=[empty_cols, empty_cols, empty_cols, empty_cols, empty_cols, empty_cols]
        )

        # Disable model, filter, buttons.
        self._combo_filters.blockSignals(True)
        self._combo_filters.clear()
        self._combo_filters.setEnabled(False)
        self._combo_filters.blockSignals(False)
        self._export_btn.setEnabled(False)

        # Setting models and columns.
        target_type = self._target_types[self._combo_target.currentText()]

        if target_type == TargetType.RESULT:
            model_ids = self._result_model_ids
            self._table.hide_columns(columns=['Prob(1)', 'Prob(X)', 'Prob(2)'], hide=False)
            self._table.hide_columns(columns=['Prob(U)', 'Prob(O)'], hide=True)

            empty_cols = ['']*10
            self._table.modify_columns(columns=['Predicted', 'Prob(U)', 'Prob(O)'], data=[empty_cols, empty_cols, empty_cols])
        elif target_type == TargetType.OVER_UNDER:
            model_ids = self._uo_model_ids
            self._table.hide_columns(columns=['Prob(1)', 'Prob(X)', 'Prob(2)'], hide=True)
            self._table.hide_columns(columns=['Prob(U)', 'Prob(O)'], hide=False)
        else:
            raise ValueError(f'Undefined targets: "{target_type}"')

        # Adding model ids.
        self._combo_model.setEnabled(True)
        self._combo_model.blockSignals(True)
        self._combo_model.clear()
        for model_id in model_ids:
            self._combo_model.addItem(model_id)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.blockSignals(False)

    def _on_model_change(self):
        pass

        fixture_df = self._read_fixture()

        if fixture_df is None:
            self._combo_model.blockSignals(True)
            self._combo_model.setCurrentIndex(-1)
            self._combo_model.blockSignals(False)
            self._combo_filters.blockSignals(True)
            self._combo_filters.clear()
            self._combo_filters.setEnabled(False)
            self._combo_filters.blockSignals(False)
            return

        self._index = fixture_df.index.to_numpy()
        self._odds = fixture_df[['1', 'X', '2']]

        # Prepare the odd mask, which is fixed.
        self._prepare_odd_mask(df=fixture_df)

        # Loading model.
        model_id = self._combo_model.currentText()
        model, model_config = self._model_db.load_model(model_id=model_id)

        # Load filters.
        self._combo_filters.blockSignals(True)
        self._combo_filters.clear()
        self._combo_filters.setEnabled(True)
        if 'eval' in model_config and 'percentiles' in model_config['eval']:
            self._combo_filters.addItem(f'--- Select Filters ---')

            self._percentiles = model_config['eval']['percentiles']
            for key in self._percentiles.keys():
                self._combo_filters.addItem(f'{key}')
        self._combo_filters.blockSignals(False)

        # Construct matches.
        df = construct_inputs_by_fixture(df=self._df, fixture_df=fixture_df)

        # Get & Store predictions.
        y_prob = model.predict_proba(df=df)
        self._y_pred = y_prob.argmax(axis=1)
        self._y_prob = y_prob.round(2)

        # Add data to table.
        target_type = self._target_types[self._combo_target.currentText()]

        if target_type == TargetType.RESULT:
            mapper = np.array(['H', 'D', 'A'])
            columns = ['Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)']
        else:
            mapper = np.array(['U', 'O'])
            columns = ['Predicted', 'Prob(U)', 'Prob(O)']

        mapped_y_pred = mapper.take(self._y_pred)
        data = np.hstack([np.expand_dims(mapped_y_pred, axis=-1), self._y_prob])
        self._table.modify_columns(columns=columns, data=data, rows=fixture_df.index.tolist())
        self._highlight_matches()

        # Enable export button.
        self._export_btn.setEnabled(True)

    def _on_filters_change(self):
        self._highlight_matches()

    def _read_fixture(self) -> Optional[pd.DataFrame]:
        """ Reads the fixture and validates the values. """

        data = []
        indices = []
        for row in range(10):
            home_item = self._table.item(row, 0)
            home = home_item.text().strip() if home_item else ""
            away_item = self._table.item(row, 1)
            away = away_item.text().strip() if away_item else ""

            if home == "" and away == "":
                continue
            elif home == "":
                QMessageBox.critical(self, 'Home Missing', f'Home team missing at row {row}.')
                return None
            elif away == "":
                QMessageBox.critical(self, 'Away Missing', f'Away team missing at row {row}.')
                return None
            elif home == away:
                QMessageBox.critical(self, 'Same Teams', f'Found matches with a single team at row {row}.')
                return None

            try:
                odd_1 = float(self._table.item(row, 2).text().strip())
                odd_x = float(self._table.item(row, 3).text().strip())
                odd_2 = float(self._table.item(row, 4).text().strip())
            except (TypeError, ValueError, AttributeError):
                QMessageBox.critical(self, 'Invalid Odds', f'Found invalid odd values or missing at row {row}.')
                return None
            else:
                if odd_1 < 1.01 or odd_x < 1.01 or odd_2 < 1.01:
                    QMessageBox.critical(self, 'Invalid Odds', f'Found odds < 1.01 at row {row}.')
                    return None

                data.append([home, away, odd_1, odd_x, odd_2])
                indices.append(row)

        fixtures_df = pd.DataFrame(data=data, columns=['Home', 'Away', '1', 'X', '2'], index=indices)
        return fixtures_df

    def _prepare_odd_mask(self, df: pd.DataFrame):
        """ Prepares the standard odd mask for this league, based on the selected odds. """

        odd_1_filter = self._league.odd_1_range
        if odd_1_filter is not None:
            min_odd, max_odd = odd_1_filter
            self._odd_mask = ((df['1'] >= min_odd) & (df['1'] <= max_odd))
        else:
            self._odd_mask = np.array([1]*df.shape[0], dtype=bool)

        odd_x_filter = self._league.odd_x_range
        if odd_x_filter is not None:
            min_odd, max_odd = odd_x_filter
            odd_x_mask = ((df['X'] >= min_odd) & (df['X'] <= max_odd))
            self._odd_mask = self._odd_mask & odd_x_mask

        odd_2_filter = self._league.odd_2_range
        if odd_2_filter is not None:
            min_odd, max_odd = odd_2_filter
            odd_2_mask = ((df['2'] >= min_odd) & (df['2'] <= max_odd))
            self._odd_mask = self._odd_mask & odd_2_mask

    def _highlight_matches(self):
        """ Filters and highlights the matches using the specified league odds and the selected filters. """

        mask = self._odd_mask
        selected_filters = self._combo_filters.getSelectedTexts()

        if selected_filters and selected_filters[0] == '--- Select Filters ---':
            selected_filters = selected_filters[1:]
        if selected_filters:
            target_type = self._target_types[self._combo_target.currentText()]
            all_filter_mask = np.zeros(shape=(mask.shape[0],), dtype=bool)
            for filter_id in selected_filters:
                # Filter odds.
                if filter_id != 'None':
                    odd, low, high = ast.literal_eval(filter_id)
                    odd_df = self._odds[odd]
                    odd_mask = (low <= odd_df) & (odd_df <= high)
                else:
                    odd_mask = np.ones(shape=(mask.shape[0],), dtype=bool)

                # Filter percentiles.
                prob_percentiles = self._percentiles[filter_id] if filter_id == 'None' else self._percentiles[ast.literal_eval(filter_id)]

                if target_type == TargetType.RESULT:
                    thresholds = np.float32([prob_percentiles['1'][1], prob_percentiles['X'][1], prob_percentiles['2'][1]])
                else:
                    thresholds = np.float32([prob_percentiles['U'][1], prob_percentiles['O'][1]])

                percentile_mask = np.all(self._y_prob >= thresholds, axis=1)
                filter_mask = odd_mask & percentile_mask
                all_filter_mask = all_filter_mask | filter_mask
            mask = mask & all_filter_mask

        highlight_ids = self._index[mask].tolist()

        if len(highlight_ids) > 0:
            self._table.highlight_rows(row_ids=highlight_ids)
        else:
            self._table.clear_selection()

    def _export(self):
        # Fetch the selected items.
        highlight_ids = sorted({index.row() for index in self._table.selectedIndexes()})

        if len(highlight_ids) == 0:
            QMessageBox.information(self, 'None Selected', 'Select the matches (rows) you want to export.')
            return

        # Export the selected items.
        data = []
        target_type = self._target_types[self._combo_target.currentText()]
        for row in range(10):
            if row in highlight_ids:
                home_item = self._table.item(row, 0)
                home = home_item.text().strip() if home_item else ""
                away_item = self._table.item(row, 1)
                away = away_item.text().strip() if away_item else ""
                odd_1 = self._table.item(row, 2).text().strip()
                odd_x = self._table.item(row, 3).text().strip()
                odd_2 = self._table.item(row, 4).text().strip()
                predicted = self._table.item(row, 5).text().strip()
                data_row = [home, away, odd_1, odd_x, odd_2, predicted]

                if target_type == TargetType.RESULT:
                    data_row.extend([
                        float(self._table.item(row, 6).text().strip()),
                        float(self._table.item(row, 7).text().strip()),
                        float(self._table.item(row, 8).text().strip())
                    ])
                else:
                    data_row.extend([
                        float(self._table.item(row, 6).text().strip()),
                        float(self._table.item(row, 7).text().strip())
                    ])

                data.append(data_row)

        if target_type == TargetType.RESULT:
            df = pd.DataFrame(data=data, columns=['Home Team', 'Away Team', '1', 'X', '2', 'Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)'])
        else:
            df = pd.DataFrame(data=data, columns=['Home Team', 'Away Team', '1', 'X', '2', 'Predicted', 'Prob(U)', 'Prob(O)'])

        default_filepath = f'{self._league.league_id}-fixures.csv'
        path, _ = QFileDialog.getSaveFileName(self, 'Export to CSV', default_filepath, 'CSV Files (*.csv)')

        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'

        file_exists = os.path.exists(path)
        try:
            if not file_exists:
                df.to_csv(path, mode='w', header=True, index=False)
            else:
                df.to_csv(path, mode='a', header=False, index=False)

            QMessageBox.information(self, 'Success', 'Export Completed!')
        except Exception as e:
            QMessageBox.critical(self, 'Export Failed', f'Could not export data.\n\nError:\n{e}')
