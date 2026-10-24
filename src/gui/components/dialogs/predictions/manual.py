from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.core.agents.environments.actions import Action
from src.core.preprocessing.utils.inputs import construct_inputs_by_match
from src.core.preprocessing.utils.targets import TargetType
from src.gui.widgets import ExcelTable

if TYPE_CHECKING:
    from src.core.database.agent import AgentDatabase
    from src.core.database.model import ModelDatabase


class ManualPredictionDialog(QDialog):
    """Build and predict one manually entered future match."""

    _BASE_COLUMNS = ['Date', 'Season', 'Week', 'Home', 'Away', '1', 'X', '2']
    _PROBABILITY_COLUMNS = ['Prob-H', 'Prob-D', 'Prob-A', 'Prob-U', 'Prob-O']
    _OUTPUT_COLUMNS = ['Predicted', *_PROBABILITY_COLUMNS, 'Betting']
    _DOUBLE_CHANCE_LABELS = ['1X', '12', 'X2']
    _METADATA_COLUMNS = {
        'Date', 'Season', 'Week', 'Home', 'Away', '1X', '12', 'X2',
        'HG', 'AG', 'Result', 'Result-U/O',
    }

    def __init__(
            self,
            df: pd.DataFrame,
            model_database: ModelDatabase,
            agent_database: AgentDatabase,
            parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._source_df = df
        self._league_df = df.dropna(ignore_index=True)
        self._model_database = model_database
        self._agent_database = agent_database
        self._feature_columns = [
            column
            for column in self._league_df.columns
            if column not in self._METADATA_COLUMNS and column not in self._BASE_COLUMNS
        ]

        self._model_ids = {target_type: [] for target_type in TargetType}
        league_models = self._model_database.index.get(self._model_database.league_id, {})
        for model_id, model_config in league_models.items():
            target_type = model_config.classifier_config.get('target_type')
            if isinstance(target_type, str):
                try:
                    target_type = TargetType(target_type)
                except ValueError:
                    continue
            if target_type in self._model_ids:
                self._model_ids[target_type].append(model_id)

        for model_ids in self._model_ids.values():
            model_ids.sort(key=str.casefold)

        self._target_probability_columns = {
            target_type: [f'Prob-{label}' for label in target_type.get_mapper()]
            for target_type in TargetType
        }
        self._prediction_df: Optional[pd.DataFrame] = None
        self._match_df: Optional[pd.DataFrame] = None
        self._loaded_models = {}
        self._loaded_agents = {}
        self._errors: dict[str, str] = {}

        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setMinimumSize(600, 400)
        self.resize(800, 600)
        self._build_ui()
        self.setWindowTitle('Manual Predictions')

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

        match_group = QGroupBox('Match', page)
        match_layout = QGridLayout(match_group)
        match_layout.setContentsMargins(12, 12, 12, 12)
        match_layout.setHorizontalSpacing(12)
        match_layout.setVerticalSpacing(10)

        self._home_team_combo = QComboBox(match_group)
        self._away_team_combo = QComboBox(match_group)
        self._populate_team_combo(self._home_team_combo, 'Home')
        self._populate_team_combo(self._away_team_combo, 'Away')

        self._odd_1_edit = QLineEdit(match_group)
        self._odd_x_edit = QLineEdit(match_group)
        self._odd_2_edit = QLineEdit(match_group)
        for odd_edit in (self._odd_1_edit, self._odd_x_edit, self._odd_2_edit):
            odd_edit.setPlaceholderText('> 1.0')
            odd_edit.setFixedWidth(90)

        match_layout.addWidget(QLabel('Home Team', match_group), 0, 0)
        match_layout.addWidget(self._home_team_combo, 0, 1)
        match_layout.addWidget(QLabel('Away Team', match_group), 0, 2)
        match_layout.addWidget(self._away_team_combo, 0, 3)
        match_layout.addWidget(QLabel('Odd 1', match_group), 1, 0)
        match_layout.addWidget(self._odd_1_edit, 1, 1)
        match_layout.addWidget(QLabel('Odd X', match_group), 1, 2)
        match_layout.addWidget(self._odd_x_edit, 1, 3)
        match_layout.addWidget(QLabel('Odd 2', match_group), 1, 4)
        match_layout.addWidget(self._odd_2_edit, 1, 5)
        match_layout.setColumnStretch(1, 1)
        match_layout.setColumnStretch(3, 1)
        page_layout.addWidget(match_group)

        prediction_group = QGroupBox('Prediction Options', page)
        prediction_layout = QGridLayout(prediction_group)
        prediction_layout.setContentsMargins(12, 12, 12, 12)
        prediction_layout.setHorizontalSpacing(12)
        prediction_layout.setVerticalSpacing(10)

        self._target_combo = QComboBox(prediction_group)
        for target_type, model_ids in self._model_ids.items():
            if model_ids:
                self._target_combo.addItem(target_type.value, userData=target_type)
        self._target_combo.setCurrentIndex(-1)
        self._target_combo.setToolTip('Selects the type of model prediction.')

        self._model_combo = QComboBox(prediction_group)
        self._model_combo.setCurrentIndex(-1)
        self._model_combo.setToolTip('Selects a trained model for the chosen target type.')

        self._agent_combo = QComboBox(prediction_group)
        self._agent_combo.addItems(sorted(self._agent_database.get_agent_ids(), key=str.casefold))
        self._agent_combo.setCurrentIndex(-1)
        self._agent_combo.setToolTip('Selects a trained agent to generate a betting action.')

        self._double_chance_checkbox = QCheckBox('Switch to Double Chance', prediction_group)
        self._double_chance_checkbox.setChecked(False)
        self._double_chance_checkbox.setEnabled(False)
        self._double_chance_checkbox.setToolTip('Convert Result probabilities into 1X, 12 and X2 probabilities.')

        prediction_layout.addWidget(QLabel('Target Type', prediction_group), 0, 0)
        prediction_layout.addWidget(self._target_combo, 0, 1)
        prediction_layout.addWidget(QLabel('Model ID', prediction_group), 0, 2)
        prediction_layout.addWidget(self._model_combo, 0, 3)
        prediction_layout.addWidget(QLabel('Agent ID', prediction_group), 1, 0)
        prediction_layout.addWidget(self._agent_combo, 1, 1, 1, 3)
        prediction_layout.addWidget(self._double_chance_checkbox, 2, 0, 1, 4)
        prediction_layout.setColumnStretch(1, 1)
        prediction_layout.setColumnStretch(3, 2)
        page_layout.addWidget(prediction_group)

        self._error_label = QLabel(page)
        self._error_label.setStyleSheet('color: red;')
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        page_layout.addWidget(self._error_label)

        table_columns = [*self._BASE_COLUMNS, *self._OUTPUT_COLUMNS, *self._feature_columns]
        table_df = pd.DataFrame([[''] * len(table_columns)], columns=table_columns, dtype=object)

        results_group = QGroupBox('Manual Prediction', page)
        results_layout = QVBoxLayout(results_group)
        self._matches_table = ExcelTable(
            parent=results_group,
            df=table_df,
            readonly=True,
            supports_sorting=False,
            supports_query_search=False,
            supports_deletion=False,
            auto_resize=True,
            allow_mouse_clicks=False,
        )
        self._matches_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._matches_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._matches_table.horizontalHeader().show()
        self._matches_table.verticalHeader().hide()
        self._matches_table.hide_columns(self._OUTPUT_COLUMNS, hide=True)
        table_height = (
            self._matches_table.horizontalHeader().sizeHint().height()
            + self._matches_table.verticalHeader().defaultSectionSize()
            + self._matches_table.horizontalScrollBar().sizeHint().height()
            + self._matches_table.frameWidth() * 2
            + 2
        )
        self._matches_table.setFixedHeight(table_height)
        results_layout.addWidget(self._matches_table)
        page_layout.addWidget(results_group)
        page_layout.addStretch(1)
        scroll_area.setWidget(page)

        self._home_team_combo.currentIndexChanged.connect(self._inputs_changed)
        self._away_team_combo.currentIndexChanged.connect(self._inputs_changed)
        self._odd_1_edit.textChanged.connect(self._inputs_changed)
        self._odd_x_edit.textChanged.connect(self._inputs_changed)
        self._odd_2_edit.textChanged.connect(self._inputs_changed)
        self._target_combo.currentIndexChanged.connect(self._target_changed)
        self._model_combo.currentIndexChanged.connect(self._inputs_changed)
        self._agent_combo.currentIndexChanged.connect(self._inputs_changed)
        self._double_chance_checkbox.toggled.connect(self._double_chance_changed)

    def _populate_team_combo(self, combo: QComboBox, column: str) -> None:
        teams = sorted(
            self._league_df[column].dropna().unique().tolist(),
            key=lambda team: str(team).casefold(),
        )
        for team in teams:
            combo.addItem(str(team), userData=team)
        combo.setCurrentIndex(-1)

    def _inputs_changed(self, _value=None) -> None:
        self._fetch_match()
        self._model_predict()
        self._agent_predict()

    def _target_changed(self, _index: int) -> None:
        target_type = self._target_combo.currentData()

        with QSignalBlocker(self._model_combo):
            self._model_combo.clear()
            if target_type is not None:
                self._model_combo.addItems(self._model_ids.get(target_type, []))
            self._model_combo.setCurrentIndex(-1)

        with QSignalBlocker(self._double_chance_checkbox):
            self._double_chance_checkbox.setChecked(False)
        self._double_chance_checkbox.setEnabled(target_type == TargetType.RESULT)

        self._configure_prediction_columns()
        self._clear_model_output()
        self._inputs_changed()

    def _double_chance_changed(self, _checked: bool) -> None:
        self._configure_prediction_columns()
        self._model_predict()

    def _configure_prediction_columns(self) -> None:
        target_type = self._target_combo.currentData()
        self._matches_table.hide_columns(['Predicted', *self._PROBABILITY_COLUMNS], hide=True)

        if target_type is None:
            self._matches_table.set_column_labels({})
            return

        probability_columns = self._target_probability_columns[target_type]
        self._matches_table.hide_columns(['Predicted', *probability_columns], hide=False)

        labels = {}
        if target_type == TargetType.RESULT and self._double_chance_checkbox.isChecked():
            labels.update({
                column: f'Prob-{label}'
                for column, label in zip(probability_columns, self._DOUBLE_CHANCE_LABELS)
            })
        self._matches_table.set_column_labels(labels)

    @staticmethod
    def _parse_odd(text: str, label: str, errors: list[str]) -> Optional[float]:
        try:
            odd = float(text.strip())
        except (TypeError, ValueError):
            errors.append(f'Odd {label} must be a number greater than 1.0.')
            return None

        if not math.isfinite(odd) or odd <= 1.0:
            errors.append(f'Odd {label} must be a number greater than 1.0.')
            return None
        return odd

    def _validate_match_fields(self) -> Optional[tuple[object, object, float, float, float]]:
        errors = []
        home = self._home_team_combo.currentData()
        away = self._away_team_combo.currentData()

        if home is None:
            errors.append('Select a home team.')
        if away is None:
            errors.append('Select an away team.')
        if home is not None and away is not None and home == away:
            errors.append('Home Team and Away Team must be different.')

        odd_1 = self._parse_odd(self._odd_1_edit.text(), '1', errors)
        odd_x = self._parse_odd(self._odd_x_edit.text(), 'X', errors)
        odd_2 = self._parse_odd(self._odd_2_edit.text(), '2', errors)

        if errors:
            self._set_error('input', ' '.join(errors))
            return None

        self._set_error('input', None)
        return home, away, odd_1, odd_x, odd_2

    def _latest_season_and_week(self) -> tuple[object, int]:
        dates = pd.to_datetime(self._source_df['Date'], errors='coerce')
        if dates.notna().any():
            latest_season = self._source_df.loc[dates.idxmax(), 'Season']
        else:
            latest_season = self._source_df['Season'].max()

        season_rows = self._source_df[self._source_df['Season'] == latest_season]
        last_week = pd.to_numeric(season_rows['Week'], errors='coerce').max()
        if pd.isna(last_week):
            raise ValueError('Could not determine the last week of the current season.')
        return latest_season, int(last_week) + 1

    def _fetch_match(self) -> None:
        self._prediction_df = None
        self._match_df = None
        self._clear_table_row()

        values = self._validate_match_fields()
        if values is None:
            return

        home, away, odd_1, odd_x, odd_2 = values
        match = pd.Series(
            [home, away, odd_1, odd_x, odd_2],
            index=['Home', 'Away', '1', 'X', '2'],
        )

        try:
            row = construct_inputs_by_match(league_df=self._league_df, match_df=match)
            if row is None:
                self._set_error('input', 'No historical match was found for the selected Home/Away pairing.')
                return

            row = row.copy()
            latest_season, next_week = self._latest_season_and_week()
            row['Date'] = datetime.now().strftime('%Y-%m-%d')
            row['Season'] = latest_season
            row['Week'] = next_week

            odd_1_probability = 1.0 / odd_1
            odd_x_probability = 1.0 / odd_x
            odd_2_probability = 1.0 / odd_2
            row['1X'] = max(round(1.0 / (odd_1_probability + odd_x_probability), 2), 1.0)
            row['12'] = max(round(1.0 / (odd_1_probability + odd_2_probability), 2), 1.0)
            row['X2'] = max(round(1.0 / (odd_x_probability + odd_2_probability), 2), 1.0)

            # Models and agents share the historical dataframe preprocessing
            # contract, which requires these metadata columns. Neutral values
            # are kept only in the private prediction input and are never shown.
            row['HG'] = 0
            row['AG'] = 0
            row[TargetType.RESULT.value] = 'D'
            row[TargetType.RESULT_UNDER_OVER.value] = 'U'
            self._prediction_df = row.to_frame().T.reset_index(drop=True)

            display_row = row.drop(
                labels=['HG', 'AG', TargetType.RESULT.value, TargetType.RESULT_UNDER_OVER.value],
                errors='ignore',
            )
            self._match_df = display_row.to_frame().T.reset_index(drop=True)
            display_columns = [*self._BASE_COLUMNS, *self._feature_columns]
            values_to_show = [display_row[column] for column in display_columns]
            self._matches_table.modify_columns(
                columns=display_columns,
                data=[values_to_show],
                rows=[0],
            )
            self._set_error('input', None)
        except Exception as error:
            self._prediction_df = None
            self._match_df = None
            self._set_error('input', f'Could not construct the match: {error}')

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
                model, _model_config = self._model_database.load_model(model_id=model_id)
                self._loaded_models[model_id] = model

            probabilities = np.asarray(model.predict_proba(df=self._prediction_df), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[0] != 1:
                raise ValueError(f'Expected one probability row, received shape {probabilities.shape}.')

            probability_columns = self._target_probability_columns[target_type]
            labels = np.asarray(list(target_type.get_mapper().keys()))
            if probabilities.shape[1] != labels.size:
                raise ValueError(
                    f'Expected {labels.size} probabilities for {target_type.value}, '
                    f'received {probabilities.shape[1]}.'
                )

            if target_type == TargetType.RESULT and self._double_chance_checkbox.isChecked():
                probabilities = np.column_stack((
                    probabilities[:, 0] + probabilities[:, 1],
                    probabilities[:, 0] + probabilities[:, 2],
                    probabilities[:, 1] + probabilities[:, 2],
                ))
                labels = np.asarray(self._DOUBLE_CHANCE_LABELS)

            prediction = labels[int(np.argmax(probabilities[0]))]
            prediction_values = [prediction, *probabilities[0].round(decimals=3).tolist()]
            self._matches_table.modify_columns(
                columns=['Predicted', *probability_columns],
                data=[prediction_values],
                rows=[0],
            )
        except Exception as error:
            self._set_error('model', f'Could not generate the model prediction: {error}')

    def _agent_predict(self) -> None:
        self._matches_table.modify_columns(columns=['Betting'], data=[['']], rows=[0])
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
            if len(actions) != 1:
                raise ValueError(f'Expected one betting action, received {len(actions)}.')
            action = Action.get_name(action=int(actions[0]))
            self._matches_table.modify_columns(columns=['Betting'], data=[[action]], rows=[0])
        except Exception as error:
            self._set_error('agent', f'Could not generate the betting prediction: {error}')

    def _clear_table_row(self) -> None:
        columns = [*self._BASE_COLUMNS, *self._OUTPUT_COLUMNS, *self._feature_columns]
        self._matches_table.modify_columns(
            columns=columns,
            data=[[''] * len(columns)],
            rows=[0],
        )

    def _clear_model_output(self) -> None:
        columns = ['Predicted', *self._PROBABILITY_COLUMNS]
        self._matches_table.modify_columns(
            columns=columns,
            data=[[''] * len(columns)],
            rows=[0],
        )

    def _set_error(self, source: str, message: Optional[str]) -> None:
        if message:
            self._errors[source] = message
        else:
            self._errors.pop(source, None)

        error_text = '\n'.join(self._errors.values())
        self._error_label.setText(error_text)
        self._error_label.setVisible(bool(error_text))
