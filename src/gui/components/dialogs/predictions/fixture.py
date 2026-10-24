from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.core.agents.environments.actions import Action
from src.core.network.fixtures.footystats import FootyStatsScraper
from src.core.preprocessing.filters import filter_matches
from src.core.preprocessing.utils.fixtures import match_fixture_teams
from src.core.preprocessing.utils.inputs import construct_inputs_by_match
from src.core.preprocessing.utils.targets import TargetType
from src.gui.components.dialogs.predictions.manual import ManualPredictionDialog
from src.gui.widgets import CheckableComboBox, ExcelTable

if TYPE_CHECKING:
    from src.core.database.agent import AgentDatabase
    from src.core.database.model import ModelDatabase


class FixturePredictionDialog(ManualPredictionDialog):
    """Download, correct, and predict every match in a FootyStats fixture."""

    _MAX_FIXTURE_ROWS = 15
    _ENTRY_COLUMNS = ['Home', 'Away', '1', 'X', '2']
    _TEAM_ERROR = 'At least one entered match is missing a Home or Away team.'
    _ODD_ERROR = 'At least one entered match has an odd that must be a number greater than 1.0.'

    def __init__(
            self,
            df: pd.DataFrame,
            fixture_url: str,
            model_database: ModelDatabase,
            agent_database: AgentDatabase,
            parent: Optional[QWidget] = None,
            scraper: Optional[FootyStatsScraper] = None,
    ):
        self._fixture_url = fixture_url
        self._scraper = scraper or FootyStatsScraper()
        self._fixture_tree = None
        self._prediction_row_ids: list[int] = []
        self._prediction_filter_df: Optional[pd.DataFrame] = None
        self._loaded_model_configs = {}
        self._selected_fixture_date = ''
        self._updating_table = False

        super().__init__(
            df=df,
            model_database=model_database,
            agent_database=agent_database,
            parent=parent,
        )

        self.setWindowTitle('Fixture Predictions')
        self._load_available_dates()

    def _build_ui(self) -> None:
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        dialog_layout.addWidget(scroll_area)

        page = QWidget(scroll_area)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(16, 16, 16, 16)
        page_layout.setSpacing(12)

        fixture_group = QGroupBox('Fixture', page)
        fixture_layout = QGridLayout(fixture_group)
        fixture_layout.setContentsMargins(12, 12, 12, 12)
        fixture_layout.setHorizontalSpacing(12)
        fixture_layout.setVerticalSpacing(10)

        self._date_combo = QComboBox(fixture_group)
        self._date_combo.setFixedWidth(150)
        self._date_combo.setCurrentIndex(-1)
        self._date_combo.setToolTip('Selects one of the available FootyStats fixture dates.')
        fixture_layout.addWidget(QLabel('Fixture Date', fixture_group), 0, 0)
        fixture_layout.addWidget(
            self._date_combo,
            0,
            1,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        fixture_layout.setColumnStretch(2, 1)
        page_layout.addWidget(fixture_group)

        prediction_group = QGroupBox('Prediction Options', page)
        prediction_layout = QGridLayout(prediction_group)
        prediction_layout.setContentsMargins(12, 12, 12, 12)
        prediction_layout.setHorizontalSpacing(12)
        prediction_layout.setVerticalSpacing(10)

        self._target_combo = QComboBox(prediction_group)
        self._target_combo.setFixedWidth(300)
        for target_type, model_ids in self._model_ids.items():
            if model_ids:
                self._target_combo.addItem(target_type.value, userData=target_type)
        self._target_combo.setCurrentIndex(-1)
        self._target_combo.setToolTip('Selects the type of model prediction.')

        self._model_combo = QComboBox(prediction_group)
        self._model_combo.setFixedWidth(300)
        self._model_combo.setCurrentIndex(-1)
        self._model_combo.setToolTip('Selects a trained model for the chosen target type.')

        self._agent_combo = QComboBox(prediction_group)
        self._agent_combo.setFixedWidth(300)
        self._agent_combo.addItems(sorted(self._agent_database.get_agent_ids(), key=str.casefold))
        self._agent_combo.setCurrentIndex(-1)
        self._agent_combo.setToolTip('Selects a trained agent to generate betting actions.')

        self._double_chance_checkbox = QCheckBox('Switch to Double Chance', prediction_group)
        self._double_chance_checkbox.setChecked(False)
        self._double_chance_checkbox.setEnabled(False)
        self._double_chance_checkbox.setToolTip('Convert Result probabilities into 1X, 12 and X2 probabilities.')

        self._saved_filters_combo = CheckableComboBox(prediction_group)
        self._saved_filters_combo.setFixedWidth(300)
        self._saved_filters_combo.setEnabled(False)
        self._saved_filters_combo.setToolTip('Select saved filters used to highlight predictions.')

        prediction_layout.addWidget(QLabel('Target Type', prediction_group), 0, 0)
        prediction_layout.addWidget(self._target_combo, 0, 1)
        prediction_layout.addWidget(QLabel('Model ID', prediction_group), 0, 2)
        prediction_layout.addWidget(self._model_combo, 0, 3)
        prediction_layout.addWidget(QLabel('Agent ID', prediction_group), 1, 0)
        prediction_layout.addWidget(self._agent_combo, 1, 1)
        prediction_layout.addWidget(QLabel('Saved Filters', prediction_group), 1, 2)
        prediction_layout.addWidget(self._saved_filters_combo, 1, 3)
        prediction_layout.addWidget(self._double_chance_checkbox, 2, 0, 1, 4)
        prediction_layout.setColumnStretch(1, 1)
        prediction_layout.setColumnStretch(3, 2)
        page_layout.addWidget(prediction_group)

        self._error_label = QLabel(page)
        self._error_label.setStyleSheet('color: red;')
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        page_layout.addWidget(self._error_label)

        self._table_columns = [*self._BASE_COLUMNS, *self._OUTPUT_COLUMNS, *self._feature_columns]
        table_df = pd.DataFrame(
            '',
            index=range(self._MAX_FIXTURE_ROWS),
            columns=self._table_columns,
            dtype=object,
        )
        teams = sorted(
            set(pd.concat([self._source_df['Home'], self._source_df['Away']]).dropna().tolist()),
            key=lambda team: str(team).casefold(),
        )

        self._results_group = QGroupBox('Fixture Predictions (0 Matches)', page)
        results_layout = QVBoxLayout(self._results_group)

        table_controls = QHBoxLayout()
        table_controls.setContentsMargins(0, 0, 0, 0)
        table_controls.setSpacing(12)

        find_replace_button = QPushButton('Find / Replace', self._results_group)
        find_replace_button.setMinimumHeight(36)
        find_replace_button.setToolTip('Find or replace values in the fixture table.')

        self._hide_features_checkbox = QCheckBox('Hide Features', self._results_group)
        self._hide_features_checkbox.setToolTip('Hide all model input feature columns.')
        self._hide_features_checkbox.setChecked(False)

        self._export_table_button = QPushButton('Export Table', self._results_group)
        self._export_table_button.setMinimumHeight(36)
        self._export_table_button.setToolTip('Export all rows in the fixture table.')

        self._export_selected_button = QPushButton('Export Selected', self._results_group)
        self._export_selected_button.setMinimumHeight(36)
        self._export_selected_button.setToolTip('Export highlighted fixture rows only.')
        self._export_selected_button.setEnabled(False)

        table_controls.addWidget(find_replace_button, alignment=Qt.AlignmentFlag.AlignVCenter)
        table_controls.addWidget(self._hide_features_checkbox, alignment=Qt.AlignmentFlag.AlignVCenter)
        table_controls.addWidget(self._export_table_button, alignment=Qt.AlignmentFlag.AlignVCenter)
        table_controls.addWidget(self._export_selected_button, alignment=Qt.AlignmentFlag.AlignVCenter)
        table_controls.addStretch(1)
        results_layout.addLayout(table_controls)

        self._matches_table = ExcelTable(
            parent=self._results_group,
            df=table_df,
            readonly=False,
            supports_sorting=False,
            supports_query_search=False,
            supports_deletion=True,
            auto_resize=True,
            suggested_values={'Home': teams, 'Away': teams},
        )
        self._matches_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._matches_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._matches_table.horizontalHeader().show()
        self._matches_table.hide_columns(self._OUTPUT_COLUMNS, hide=True)
        find_replace_button.clicked.connect(self._matches_table.open_find_dialog)
        self._export_table_button.clicked.connect(self._matches_table.export_table)
        self._export_selected_button.clicked.connect(self._matches_table.export_selected)
        self._matches_table.selectionModel().selectionChanged.connect(
            self._update_export_selected_button
        )
        self._hide_features_checkbox.toggled.connect(
            lambda checked: self._matches_table.hide_columns(self._feature_columns, hide=checked)
        )
        results_table_height = (
            self._matches_table.horizontalHeader().sizeHint().height()
            + self._matches_table.verticalHeader().defaultSectionSize() * self._MAX_FIXTURE_ROWS
            + self._matches_table.horizontalScrollBar().sizeHint().height()
            + self._matches_table.frameWidth() * 2
            + 2
        )
        self._matches_table.setMinimumHeight(results_table_height)
        results_layout.addWidget(self._matches_table)
        page_layout.addWidget(self._results_group, 1)
        scroll_area.setWidget(page)

        self._date_combo.currentIndexChanged.connect(self._date_changed)
        self._target_combo.currentIndexChanged.connect(self._target_changed)
        self._model_combo.currentIndexChanged.connect(self._model_changed)
        self._agent_combo.currentIndexChanged.connect(self._inputs_changed)
        self._double_chance_checkbox.toggled.connect(self._double_chance_changed)
        self._saved_filters_combo.checkedItemsChanged.connect(self._saved_filter_changed)
        self._matches_table.model().dataChanged.connect(self._table_data_changed)

    def _update_export_selected_button(self, *_args) -> None:
        self._export_selected_button.setEnabled(self._matches_table.has_highlighted_rows)

    def _load_available_dates(self) -> None:
        with QSignalBlocker(self._date_combo):
            self._date_combo.clear()
            self._date_combo.setCurrentIndex(-1)

        try:
            self._fixture_tree = self._scraper.scrape(fixture_url=self._fixture_url)
            dates = self._scraper.get_available_table_dates(tree=self._fixture_tree)
            if not dates:
                self._set_error('download', 'No fixture dates were found on FootyStats.')
                self._date_combo.setEnabled(False)
                return

            dates = self._add_years_to_dates(date_items=dates)
            with QSignalBlocker(self._date_combo):
                self._date_combo.addItems([str(date) for date in dates])
                self._date_combo.setCurrentIndex(-1)
            self._date_combo.setEnabled(True)
            self._set_error('download', None)
        except Exception as error:
            self._fixture_tree = None
            self._date_combo.setEnabled(False)
            self._set_error('download', f'Could not download fixture dates: {error}')

    @staticmethod
    def _add_years_to_dates(
            date_items: list[str],
            current_date: Optional[date] = None,
    ) -> list[str]:
        """Add years to fixture labels ordered from newest to oldest."""

        today = current_date or date.today()
        parsed_dates = []
        for date_item in date_items:
            normalized_date = re.sub(
                r'(?<=\d)(?:st|nd|rd|th)\b',
                '',
                str(date_item).strip(),
                flags=re.IGNORECASE,
            )
            parsed_date = pd.to_datetime(normalized_date, errors='coerce')
            parsed_dates.append((normalized_date, parsed_date))

        first_valid_date = next(
            (parsed_date for _, parsed_date in parsed_dates if not pd.isna(parsed_date)),
            None,
        )
        if first_valid_date is None:
            return [normalized_date for normalized_date, _ in parsed_dates]

        first_month_day = (first_valid_date.month, first_valid_date.day)
        current_month_day = (today.month, today.day)
        inferred_year = today.year if first_month_day >= current_month_day else today.year + 1
        previous_date: Optional[date] = None
        dated_labels = []

        for normalized_date, parsed_date in parsed_dates:
            if pd.isna(parsed_date):
                dated_labels.append(normalized_date)
                continue

            item_date = date(inferred_year, parsed_date.month, parsed_date.day)
            while previous_date is not None and item_date > previous_date:
                inferred_year -= 1
                item_date = date(inferred_year, parsed_date.month, parsed_date.day)

            dated_labels.append(f'{normalized_date}, {inferred_year:04d}')
            previous_date = item_date

        return dated_labels

    def _date_changed(self, index: int) -> None:
        self._set_error('input', None)
        if index < 0:
            self._selected_fixture_date = ''
            self._replace_fixture_table(pd.DataFrame(columns=self._table_columns, dtype=object))
            self._prediction_df = None
            self._prediction_filter_df = None
            self._match_df = None
            self._prediction_row_ids = []
            self._matches_table.clearSelection()
            self._set_error('model', None)
            self._set_error('agent', None)
            return

        self._selected_fixture_date = self._format_fixture_date(self._date_combo.currentText())
        self._fetch_fixture(index=index)
        self._model_predict()
        self._agent_predict()

    @staticmethod
    def _format_fixture_date(value: str) -> str:
        normalized_value = re.sub(
            r'(?<=\d)(?:st|nd|rd|th)\b',
            '',
            value.strip(),
            flags=re.IGNORECASE,
        )
        parsed_date = pd.to_datetime(normalized_value, errors='coerce')
        if pd.isna(parsed_date):
            return value

        # FootyStats headings commonly omit the year. Pandas represents such
        # values with year 1, so attach the current calendar year explicitly.
        year = parsed_date.year if parsed_date.year > 1 else datetime.now().year
        return f'{year:04d}-{parsed_date.month:02d}-{parsed_date.day:02d}'

    def _fetch_fixture(self, index: int) -> None:
        try:
            fixture_df = self._scraper.parse_league_table(
                tree=self._fixture_tree,
                table_index=index,
            )
            if fixture_df is None or fixture_df.empty:
                self._replace_fixture_table(pd.DataFrame(columns=self._table_columns, dtype=object))
                self._prediction_df = None
                self._match_df = None
                self._prediction_row_ids = []
                self._set_error('download', 'No matches were found for the selected fixture date.')
                return

            fixture_df = self._trim_fixture_rows(fixture_df)
            if fixture_df.empty:
                self._replace_fixture_table(pd.DataFrame(columns=self._table_columns, dtype=object))
                self._prediction_df = None
                self._match_df = None
                self._prediction_row_ids = []
                self._set_error('download', 'No populated matches were found for the selected fixture date.')
                return

            fixture_df = match_fixture_teams(
                league_df=self._source_df,
                fixture_df=fixture_df.copy(),
            )
            self._set_error('download', None)
            self._construct_fixture_rows(fixture_df[self._ENTRY_COLUMNS].copy())
        except Exception as error:
            self._replace_fixture_table(pd.DataFrame(columns=self._table_columns, dtype=object))
            self._prediction_df = None
            self._match_df = None
            self._prediction_row_ids = []
            self._set_error('download', f'Could not download the selected fixture: {error}')

    def _fetch_match(self) -> None:
        """Rebuild model inputs from the fixture's currently edited cells."""

        if self._matches_table.dataframe.empty:
            self._prediction_df = None
            self._prediction_row_ids = []
            return

        fixture_df = self._matches_table.dataframe[self._ENTRY_COLUMNS].copy()
        self._construct_fixture_rows(fixture_df=fixture_df)

    def _construct_fixture_rows(self, fixture_df: pd.DataFrame) -> None:
        fixture_df = self._trim_fixture_rows(fixture_df)
        errors: list[str] = []
        has_team_error = False
        has_odd_error = False
        prediction_rows = []
        prediction_row_ids = []
        table_rows = []

        try:
            latest_season, next_week = self._latest_season_and_week()
        except Exception as error:
            self._set_error('input', str(error))
            return

        for table_row, fixture in fixture_df.reset_index(drop=True).iterrows():
            row_number = table_row + 1
            home = fixture['Home']
            away = fixture['Away']
            odd_1 = self._fixture_odd(fixture['1'])
            odd_x = self._fixture_odd(fixture['X'])
            odd_2 = self._fixture_odd(fixture['2'])
            has_odd_error = has_odd_error or any(odd is None for odd in (odd_1, odd_x, odd_2))

            values = {column: '' for column in self._table_columns}
            values.update({
                'Date': self._selected_fixture_date,
                'Season': latest_season,
                'Week': next_week,
                'Home': '' if pd.isna(home) else home,
                'Away': '' if pd.isna(away) else away,
                '1': fixture['1'],
                'X': fixture['X'],
                '2': fixture['2'],
            })

            teams_valid = True
            if pd.isna(home) or str(home).strip() == '':
                has_team_error = True
                teams_valid = False
            if pd.isna(away) or str(away).strip() == '':
                has_team_error = True
                teams_valid = False
            if teams_valid and home == away:
                errors.append(f'Row {row_number}: Home and Away teams must be different.')
                teams_valid = False

            prediction_row = None
            if teams_valid and odd_1 is not None and odd_x is not None and odd_2 is not None:
                match = pd.Series(
                    [home, away, odd_1, odd_x, odd_2],
                    index=['Home', 'Away', '1', 'X', '2'],
                )
                construction_failed = False
                try:
                    prediction_row = construct_inputs_by_match(
                        league_df=self._league_df,
                        match_df=match,
                    )
                except Exception as error:
                    construction_failed = True
                    prediction_row = None
                    errors.append(f'Row {row_number}: could not construct model inputs: {error}')
                if prediction_row is None and not construction_failed:
                    errors.append(
                        f'Row {row_number}: no historical match was found for {home} vs {away}.'
                    )

            if prediction_row is not None:
                prediction_row = prediction_row.copy()
                prediction_row['Date'] = self._selected_fixture_date
                prediction_row['Season'] = latest_season
                prediction_row['Week'] = next_week
                prediction_row['1X'] = self._double_chance_odd(odd_1, odd_x)
                prediction_row['12'] = self._double_chance_odd(odd_1, odd_2)
                prediction_row['X2'] = self._double_chance_odd(odd_x, odd_2)
                prediction_row['HG'] = 0
                prediction_row['AG'] = 0
                prediction_row[TargetType.RESULT.value] = 'D'
                prediction_row[TargetType.RESULT_UNDER_OVER.value] = 'U'

                values.update({column: prediction_row[column] for column in self._BASE_COLUMNS})
                values.update({column: prediction_row[column] for column in self._feature_columns})
                prediction_rows.append(prediction_row)
                prediction_row_ids.append(table_row)

            table_rows.append([values[column] for column in self._table_columns])

        table_df = pd.DataFrame(table_rows, columns=self._table_columns, dtype=object)
        self._replace_fixture_table(table_df)
        self._match_df = table_df[[*self._BASE_COLUMNS, *self._feature_columns]].copy()
        self._prediction_row_ids = prediction_row_ids
        self._prediction_df = (
            pd.DataFrame(prediction_rows).reset_index(drop=True)
            if prediction_rows
            else None
        )
        validation_errors = []
        if has_team_error:
            validation_errors.append(self._TEAM_ERROR)
        if has_odd_error:
            validation_errors.append(self._ODD_ERROR)
        validation_errors.extend(errors)
        self._set_error('input', ' '.join(validation_errors) if validation_errors else None)

    @staticmethod
    def _fixture_odd(value) -> Optional[float]:
        try:
            odd = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(odd) or odd <= 1.0:
            return None
        return odd

    @classmethod
    def _trim_fixture_rows(cls, fixture_df: pd.DataFrame) -> pd.DataFrame:
        """Drop fully empty entry rows and enforce the agent-compatible limit."""

        if fixture_df.empty:
            return fixture_df.reset_index(drop=True)

        def has_entry(row: pd.Series) -> bool:
            return any(
                not pd.isna(value) and str(value).strip() != ''
                for value in row
            )

        populated_mask = fixture_df[cls._ENTRY_COLUMNS].apply(has_entry, axis=1)
        return (
            fixture_df.loc[populated_mask]
            .head(cls._MAX_FIXTURE_ROWS)
            .reset_index(drop=True)
        )

    @staticmethod
    def _double_chance_odd(first_odd: float, second_odd: float) -> float:
        return max(round(1.0 / ((1.0 / first_odd) + (1.0 / second_odd)), 2), 1.0)

    def _replace_fixture_table(self, table_df: pd.DataFrame) -> None:
        populated_count = self._trim_fixture_rows(table_df).shape[0]
        table_df = (
            table_df.head(self._MAX_FIXTURE_ROWS)
            .reset_index(drop=True)
            .reindex(range(self._MAX_FIXTURE_ROWS), fill_value='')
        )
        self._updating_table = True
        try:
            self._matches_table.replace_dataframe(table_df)
        finally:
            self._updating_table = False
        self._results_group.setTitle(f'Fixture Predictions ({populated_count} Matches)')

    def _target_changed(self, index: int) -> None:
        self._prediction_filter_df = None
        self._clear_saved_filters()
        super()._target_changed(index)

    def _model_changed(self, _index: int) -> None:
        self._prediction_filter_df = None
        self._matches_table.clearSelection()
        self._load_saved_filters()
        self._inputs_changed()

    def _double_chance_changed(self, checked: bool) -> None:
        self._prediction_filter_df = None
        self._load_saved_filters()
        super()._double_chance_changed(checked)

    def _clear_saved_filters(self) -> None:
        with QSignalBlocker(self._saved_filters_combo):
            self._saved_filters_combo.clear()
        self._saved_filters_combo.setEnabled(False)
        self._set_error('filter', None)

    def _load_saved_filters(self) -> None:
        """Load saved filters for the selected prediction mode."""

        model_id = self._model_combo.currentText()
        model_config = self._loaded_model_configs.get(model_id)
        if model_config is None and model_id:
            model_config = self._model_database.index.get(
                self._model_database.league_id,
                {},
            ).get(model_id)

        use_double_chance = self._double_chance_checkbox.isChecked()
        available_filters = (
            []
            if model_config is None
            else list(getattr(model_config, 'eval_filters', None) or [])
        )
        matching_filters = [
            eval_filter
            for eval_filter in available_filters
            if eval_filter.double_chance == use_double_chance
        ]

        with QSignalBlocker(self._saved_filters_combo):
            self._saved_filters_combo.clear()
            for eval_filter in matching_filters:
                self._saved_filters_combo.addItem(
                    self._format_saved_filter(eval_filter),
                    userData=eval_filter,
                )

        self._saved_filters_combo.setEnabled(bool(matching_filters))
        self._set_error('filter', None)

    def _format_saved_filter(self, eval_filter) -> str:
        if eval_filter.prediction_filter is None:
            prediction_text = 'None'
        else:
            target_type = self._target_combo.currentData()
            prediction_labels = (
                self._DOUBLE_CHANCE_LABELS
                if self._double_chance_checkbox.isChecked()
                else list(target_type.get_mapper().keys()) if target_type is not None else []
            )
            prediction_index = eval_filter.prediction_filter
            prediction_text = (
                prediction_labels[prediction_index]
                if 0 <= prediction_index < len(prediction_labels)
                else str(prediction_index)
            )

        if eval_filter.odd_filter is None:
            odd_text = 'None'
        else:
            odd, minimum, maximum = eval_filter.odd_filter
            odd_text = f'{odd}:({round(minimum, 1)}-{round(maximum, 1)})'

        if eval_filter.percentile_filter is None:
            percentile_text = 'None'
        else:
            target, minimum, percentile = eval_filter.percentile_filter
            percentile_text = f'{target}:({round(minimum, 2)},{percentile})'

        return f'{prediction_text} | {odd_text} | {percentile_text}'

    def _selected_saved_filters(self) -> list:
        model = self._saved_filters_combo.model()
        return [
            self._saved_filters_combo.itemData(index)
            for index in range(self._saved_filters_combo.count())
            if model.item(index).checkState() == Qt.CheckState.Checked
        ]

    def _saved_filter_changed(self) -> None:
        self._highlight_predictions()

    def _inputs_changed(self, _value=None) -> None:
        self._fetch_match()
        self._model_predict()
        self._agent_predict()

    def _table_data_changed(self, top_left, bottom_right, _roles=None) -> None:
        if self._updating_table or not top_left.isValid() or not bottom_right.isValid():
            return

        changed_columns = {
            self._table_columns[column]
            for column in range(top_left.column(), bottom_right.column() + 1)
        }
        fixture_columns = {'Home', 'Away', '1', 'X', '2'}
        if changed_columns & fixture_columns:
            self._inputs_changed()
            return

        synchronizable_columns = set(self._BASE_COLUMNS + self._feature_columns) - fixture_columns
        columns_to_sync = changed_columns & synchronizable_columns
        if not columns_to_sync or self._prediction_df is None:
            return

        table_df = self._matches_table.dataframe
        prediction_positions = {
            table_row: position
            for position, table_row in enumerate(self._prediction_row_ids)
        }
        for table_row in range(top_left.row(), bottom_right.row() + 1):
            prediction_position = prediction_positions.get(table_row)
            if prediction_position is None:
                continue
            for column in columns_to_sync:
                self._prediction_df.at[prediction_position, column] = table_df.at[table_row, column]

        self._model_predict()
        self._agent_predict()

    def _clear_model_output(self) -> None:
        self._prediction_filter_df = None
        self._matches_table.clearSelection()
        columns = ['Predicted', *self._PROBABILITY_COLUMNS]
        row_count = self._matches_table.dataframe.shape[0]
        if row_count:
            self._matches_table.modify_columns(
                columns=columns,
                data=[[''] * len(columns) for _ in range(row_count)],
                rows=list(range(row_count)),
            )

    def _model_predict(self) -> None:
        self._clear_model_output()
        self._set_error('model', None)

        target_type = self._target_combo.currentData()
        model_id = self._model_combo.currentText()
        if self._prediction_df is None or target_type is None or not model_id:
            return

        try:
            model = self._loaded_models.get(model_id)
            if model is None:
                model, model_config = self._model_database.load_model(model_id=model_id)
                self._loaded_models[model_id] = model
                self._loaded_model_configs[model_id] = model_config
                self._load_saved_filters()

            raw_probabilities = np.asarray(model.predict_proba(df=self._prediction_df), dtype=float)
            expected_rows = len(self._prediction_row_ids)
            if raw_probabilities.ndim != 2 or raw_probabilities.shape[0] != expected_rows:
                raise ValueError(
                    f'Expected {expected_rows} probability rows, received shape {raw_probabilities.shape}.'
                )

            probability_columns = self._target_probability_columns[target_type]
            raw_labels = np.asarray(list(target_type.get_mapper().keys()))
            if raw_probabilities.shape[1] != raw_labels.size:
                raise ValueError(
                    f'Expected {raw_labels.size} probabilities for {target_type.value}, '
                    f'received {raw_probabilities.shape[1]}.'
                )

            probabilities = raw_probabilities
            labels = raw_labels
            if target_type == TargetType.RESULT and self._double_chance_checkbox.isChecked():
                probabilities = np.column_stack((
                    raw_probabilities[:, 0] + raw_probabilities[:, 1],
                    raw_probabilities[:, 0] + raw_probabilities[:, 2],
                    raw_probabilities[:, 1] + raw_probabilities[:, 2],
                ))
                labels = np.asarray(self._DOUBLE_CHANCE_LABELS)

            prediction_ids = np.argmax(probabilities, axis=1)
            predictions = labels[prediction_ids]
            rounded_probabilities = probabilities.round(decimals=3)
            prediction_data = [
                [prediction, *probability_row.tolist()]
                for prediction, probability_row in zip(predictions, rounded_probabilities)
            ]
            self._matches_table.modify_columns(
                columns=['Predicted', *probability_columns],
                data=prediction_data,
                rows=self._prediction_row_ids,
            )

            filter_df = self._prediction_df.copy().reset_index(drop=True)
            for label, values in zip(raw_labels, raw_probabilities.T):
                filter_df[f'Prob-{label}'] = values
            if target_type == TargetType.RESULT and self._double_chance_checkbox.isChecked():
                for label, values in zip(self._DOUBLE_CHANCE_LABELS, probabilities.T):
                    filter_df[f'Prob-{label}'] = values
            filter_df['Prediction'] = prediction_ids
            self._prediction_filter_df = filter_df
            self._highlight_predictions()
        except Exception as error:
            self._prediction_filter_df = None
            self._matches_table.clearSelection()
            self._set_error('model', f'Could not generate model predictions: {error}')

    def _highlight_predictions(self) -> None:
        """Highlight predicted rows, narrowed by checked saved filters."""

        if self._prediction_filter_df is None:
            self._matches_table.clearSelection()
            self._set_error('filter', None)
            return

        try:
            selected_filters = self._selected_saved_filters()
            filtered_df = (
                filter_matches(
                    matches_df=self._prediction_filter_df,
                    eval_filters=selected_filters,
                )
                if selected_filters
                else self._prediction_filter_df
            )
            highlighted_rows = [
                self._prediction_row_ids[position]
                for position in filtered_df.index
                if 0 <= position < len(self._prediction_row_ids)
            ]
            self._matches_table.highlight_rows(row_ids=highlighted_rows)
            self._set_error('filter', None)
        except Exception as error:
            self._matches_table.clearSelection()
            self._set_error('filter', f'Could not apply the selected saved filters: {error}')

    def _agent_predict(self) -> None:
        row_count = self._matches_table.dataframe.shape[0]
        if row_count:
            self._matches_table.modify_columns(
                columns=['Betting'],
                data=[[''] for _ in range(row_count)],
                rows=list(range(row_count)),
            )
        self._set_error('agent', None)

        agent_id = self._agent_combo.currentText()
        self._matches_table.hide_columns('Betting', hide=not bool(agent_id))
        if self._prediction_df is None or not agent_id:
            return

        try:
            agent = self._loaded_agents.get(agent_id)
            if agent is None:
                agent, _agent_config = self._agent_database.load_agent(agent_id=agent_id)
                self._loaded_agents[agent_id] = agent

            actions, _reward, _metrics = agent.compute_actions(df=self._prediction_df)
            if len(actions) != len(self._prediction_row_ids):
                raise ValueError(
                    f'Expected {len(self._prediction_row_ids)} betting actions, received {len(actions)}.'
                )
            action_data = [[Action.get_name(action=int(action))] for action in actions]
            self._matches_table.modify_columns(
                columns=['Betting'],
                data=action_data,
                rows=self._prediction_row_ids,
            )
        except Exception as error:
            self._set_error('agent', f'Could not generate betting predictions: {error}')
