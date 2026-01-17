import math
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QDialog, QFrame, QLabel, QComboBox, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QSlider
from superqt import QLabeledSlider
from src.database.model import ModelDatabase
from src.gui.utils.taskrunner import TaskRunnerDialog
from src.gui.widgets.tables import ExcelTable
from src.metrics.balance import compute_profit_balance
from src.preprocessing.utils.target import TargetType
from src.preprocessing.utils.target import construct_targets


class EvaluatorDialog(QDialog):
    """ Evaluator dialog which re-evaluates the model in the specified data. """

    def __init__(self, df: pd.DataFrame, model_db: ModelDatabase):
        super().__init__()

        self._df = df.reset_index(drop=True)

        self._model_db = model_db
        self._model_ids = model_db.get_model_ids()
        self._title = 'Evaluation Dialog'
        self._width = 800
        self._height = 600

        # Declare placeholders.
        self._model = None
        self._model_config = None
        self._percentiles = None
        self._num_eval_samples = None
        self._num_train_samples = None
        self._y_pred = None
        self._y_prob = None
        self._prob_percentiles = None
        self._correct_mask = None

        self._dataset_mask_dict = {'All': np.array([True]*self._df.shape[0], dtype=bool), 'Train': None, 'Eval': None}
        self._y_true_dict = {
            TargetType.RESULT: construct_targets(df=self._df, target_type=TargetType.RESULT),
            TargetType.OVER_UNDER: construct_targets(df=self._df, target_type=TargetType.OVER_UNDER)
        }
        self._target_types = {'Result (1/X/2)': TargetType.RESULT, 'U/O-2.5': TargetType.OVER_UNDER}
        self._datasets = ['All', 'Train', 'Eval']
        self._odd_ranges = [
            'None',
            ('1', 1.00, 1.3), ('1', 1.31, 1.6), ('1', 1.61, 1.9), ('1', 1.91, 2.5), ('1', 2.5, 3.5), ('1', 3.51, 100),
            ('X', 1.00, 2.0), ('X', 2.0, 3.0), ('X', 3.01, 100),
            ('2', 1.00, 1.3), ('2', 1.31, 1.6), ('2', 1.61, 1.9), ('2', 1.91, 2.5), ('2', 2.5, 3.5), ('2', 3.51, 100)
        ]
        self._result_model_ids = []
        self._uo_model_ids = []
        for model_id in self._model_ids:
            config = model_db.load_model_config(model_id=model_id)
            if config['target_type'] == TargetType.RESULT:
                self._result_model_ids.append(model_id)
            else:
                self._uo_model_ids.append(model_id)

        # Declare UI Placeholders.
        self._combo_model = None
        self._combo_target = None
        self._combo_dataset = None
        self._spin_eval_samples = None
        self._combo_range = None
        self._slider_percentile_1 = None
        self._slider_percentile_X = None
        self._slider_percentile_2 = None
        self._slider_percentile_under = None
        self._slider_percentile_over = None
        self._table = None
        self._store_filters_btn = None
        self._acc_label = None
        self._f1_label = None
        self._prec_label = None
        self._rec_label = None
        self._samples_label = None
        self._profit_balance_label = None

        self._initialize_window()
        self._add_widgets()

    def exec(self):
        if len(self._model_ids) == 0:
            QMessageBox.critical(
                self,
                'No Existing Models.',
                'There are no existing models to evaluate.',
                QMessageBox.StandardButton.Ok
            )
            return QDialog.Rejected

        QTimer.singleShot(0, self._show_instructions)
        super().exec()

    def _initialize_window(self):
        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)

    def _add_widgets(self):
        root = QVBoxLayout(self)

        # --- Model initialization ---
        model_hbox = QHBoxLayout()
        model_hbox.addStretch(1)

        self._combo_target = QComboBox()
        self._combo_target.setFixedWidth(120)
        for target in self._target_types:
            self._combo_target.addItem(target)
        self._combo_target.setCurrentIndex(-1)
        self._combo_target.currentIndexChanged.connect(self._on_target_change)
        model_hbox.addWidget(QLabel(' Target: '))
        model_hbox.addWidget(self._combo_target)

        self._combo_model = QComboBox()
        self._combo_model.setFixedWidth(220)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.currentIndexChanged.connect(self._on_model_change)
        model_hbox.addWidget(QLabel(' Model ID: '))
        model_hbox.addWidget(self._combo_model)

        self._combo_dataset = QComboBox()
        self._combo_dataset.setFixedWidth(120)
        for dataset_type in self._datasets:
            self._combo_dataset.addItem(dataset_type)
        self._combo_dataset.currentIndexChanged.connect(self._on_dataset_change)
        self._combo_dataset.setEnabled(False)
        model_hbox.addWidget(QLabel(' Dataset: '))
        model_hbox.addWidget(self._combo_dataset)

        model_hbox.addStretch(1)
        root.addLayout(model_hbox)

        # --- Percentile Filters ---
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
        row.addWidget(QLabel('Percentile Filters'))
        row.addWidget(right_line, 1)
        row.addStretch(1)
        root.addLayout(row)

        filters_hbox = QHBoxLayout()
        filters_hbox.addStretch(1)

        self._combo_range = QComboBox()
        self._combo_range.setFixedWidth(120)
        self._combo_range.addItem('None')
        for odd_range in self._odd_ranges[1:]:
            odd, minval, maxval = odd_range
            self._combo_range.addItem(f'{odd}-[{minval}, {maxval}]')
        self._combo_range.currentIndexChanged.connect(self._on_range_change)
        filters_hbox.addWidget(QLabel(' Odd Range: '))
        filters_hbox.addWidget(self._combo_range)

        # Add odds filters here (3 center left, 2 center right)

        filters_hbox.addStretch(1)
        root.addLayout(filters_hbox)

        def make_percentile_slider(slider_row, name: str) -> QLabeledSlider:
            slider = QLabeledSlider(Qt.Orientation.Horizontal)
            slider.setFixedWidth(180)
            slider.setRange(0, 100)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(10)
            slider.setSingleStep(1)
            slider_row.addWidget(QLabel(f'P-{name}'))
            slider_row.addWidget(slider)
            slider.setTracking(False)
            slider.valueChanged.connect(self._on_percentile_change)
            return slider

        result_hbox = QHBoxLayout()
        result_hbox.addStretch(1)
        self._slider_percentile_1 = make_percentile_slider(slider_row=result_hbox, name='Home')
        self._slider_percentile_X = make_percentile_slider(slider_row=result_hbox, name='Draw')
        self._slider_percentile_2 = make_percentile_slider(slider_row=result_hbox, name='Away')
        result_hbox.addStretch(1)
        root.addLayout(result_hbox)

        uo_hbox = QHBoxLayout()
        uo_hbox.addStretch(1)
        self._slider_percentile_under = make_percentile_slider(slider_row=uo_hbox, name='Under')
        self._slider_percentile_over = make_percentile_slider(slider_row=uo_hbox, name='Over')
        uo_hbox.addStretch(1)
        root.addLayout(uo_hbox)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self._store_filters_btn = QPushButton('Store Filters')
        self._store_filters_btn.setFixedWidth(120)
        self._store_filters_btn.setFixedHeight(30)
        self._store_filters_btn.setEnabled(False)
        self._store_filters_btn.clicked.connect(self._store_filters)
        btn_row.addWidget(self._store_filters_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        metrics_row = QHBoxLayout()
        metrics_row.addStretch(1)
        metrics_row.setSpacing(20)
        font = self.font()
        font.setPointSize(12)
        self._acc_label = QLabel('Accuracy: 0.0')
        self._acc_label.setFont(font)
        self._f1_label = QLabel('F1:  0.0')
        self._f1_label.setFont(font)
        self._prec_label = QLabel('Precision:  0.0')
        self._prec_label.setFont(font)
        self._rec_label = QLabel('Recall:  0.0')
        self._rec_label.setFont(font)
        self._samples_label = QLabel(f'Samples: {self._df.shape[0]}')
        self._samples_label.setFont(font)
        self._profit_balance_label = QLabel('Prof. Balance: 0.0')
        self._profit_balance_label.setFont(font)
        metrics_row.addWidget(self._acc_label)
        metrics_row.addWidget(self._f1_label)
        metrics_row.addWidget(self._prec_label)
        metrics_row.addWidget(self._rec_label)
        metrics_row.addWidget(self._samples_label)
        metrics_row.addWidget(self._profit_balance_label)
        metrics_row.addStretch(1)
        root.addLayout(metrics_row)

        table_df = self._df[['Date', 'Season', 'Week', 'Home', 'Away', '1', 'X', '2', 'Result', 'Result-U/O']]
        table_df[['Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)', 'Prob(U)', 'Prob(O)']] = ''
        self._table = ExcelTable(parent=self, df=table_df, readonly=True, supports_sorting=False, supports_query_search=True)
        root.addWidget(self._table)

    def _on_target_change(self):
        target_type = self._target_types[self._combo_target.currentText()]
        self._add_model_ids(target_type=target_type)
        self._adjust_table_columns(target_type=target_type)
        self._reset_evaluator_state()

    def _on_model_change(self):
        self._load_model()
        self._predict()

        self._set_percentile_states()
        self._update_prob_percentiles()

        self._update_table_predictions()
        self._update_table_and_metrics()
        self._store_filters_btn.setEnabled(True)
        self._combo_dataset.setEnabled(True)

    def _on_dataset_change(self):
        self._update_prob_percentiles()
        self._update_table_and_metrics()

    def _on_range_change(self):
        self._set_percentile_states()
        self._update_prob_percentiles()
        self._update_table_and_metrics()

    def _on_percentile_change(self):
        self._update_prob_percentiles()
        self._update_table_and_metrics()

    def _store_filters(self):
        """ Stores the selected filter (range-percentiles) pairs in config. """

        if self._percentiles is None:
            self._percentiles = {}

        odd_range = self._odd_ranges[self._combo_range.currentIndex()]
        self._percentiles[odd_range] = self._prob_percentiles

        # Store filters.
        if 'eval' not in self._model_config:
            self._model_config['eval'] = {}

        self._model_config['eval']['percentiles'] = self._percentiles
        self._model_db.update_model_config(model_config=self._model_config)

    def _reset_evaluator_state(self):
        self._model = self._model_config = self._percentiles = None
        self._num_eval_samples = self._num_train_samples = self._y_pred = self._y_prob = None
        self._y_pred = self._y_prob = self._prob_percentiles = self._correct_ids_set = self._correct_mask = None
        self._dataset_mask_dict.update({'Train': None, 'Eval': None})

        self._combo_dataset.setEnabled(False)
        self._combo_dataset.setCurrentIndex(0)
        self._set_percentile_states()
        self._table.clear_selection()
        self._update_table_predictions()
        self._update_table_and_metrics()

    def _add_model_ids(self, target_type:  TargetType):
        """ Adds model ids based on the selected target. """

        # Clearing model ids.
        self._combo_model.blockSignals(True)
        self._combo_model.clear()

        # Setting model. percentiles.
        if target_type == TargetType.RESULT:
            model_ids = self._result_model_ids
            self._slider_percentile_1.setEnabled(True)
            self._slider_percentile_X.setEnabled(True)
            self._slider_percentile_2.setEnabled(True)
            self._slider_percentile_under.setEnabled(False)
            self._slider_percentile_over.setEnabled(False)
        elif target_type == TargetType.OVER_UNDER:
            model_ids = self._uo_model_ids
            self._slider_percentile_1.setEnabled(False)
            self._slider_percentile_X.setEnabled(False)
            self._slider_percentile_2.setEnabled(False)
            self._slider_percentile_under.setEnabled(True)
            self._slider_percentile_over.setEnabled(True)
        else:
            raise ValueError(f'Undefined targets: "{target_type}"')

        # Adding model ids.
        for model_id in model_ids:
            self._combo_model.addItem(model_id)
        self._combo_model.setCurrentIndex(-1)
        self._combo_model.blockSignals(False)

    def _adjust_table_columns(self, target_type: TargetType):
        """ Adjusts table columns based on the selected target. """

        if target_type == TargetType.RESULT:
            self._table.hide_columns(columns=['Result', 'Prob(1)', 'Prob(X)', 'Prob(2)'], hide=False)
            self._table.hide_columns(columns=['Result-U/O', 'Prob(U)', 'Prob(O)'], hide=True)
        elif target_type == TargetType.OVER_UNDER:
            self._table.hide_columns(columns=['Result', 'Prob(1)', 'Prob(X)', 'Prob(2)'], hide=True)
            self._table.hide_columns(columns=['Result-U/O', 'Prob(U)', 'Prob(O)'], hide=False)
        else:
            raise ValueError(f'Undefined targets: "{target_type}"')

    def _set_percentile_states(self):
        """ Restores/Resets the model output probabilities percentiles. """

        def block_percentile_signal(block: bool):
            self._slider_percentile_1.blockSignals(block)
            self._slider_percentile_X.blockSignals(block)
            self._slider_percentile_2.blockSignals(block)
            self._slider_percentile_under.blockSignals(block)
            self._slider_percentile_over.blockSignals(block)

        def set_values(val_1: int, val_x: int, val_2: int, val_u: int, val_o: int):
            block_percentile_signal(block=True)
            self._slider_percentile_1.setValue(val_1)
            self._slider_percentile_X.setValue(val_x)
            self._slider_percentile_2.setValue(val_2)
            self._slider_percentile_under.setValue(val_u)
            self._slider_percentile_over.setValue(val_o)
            block_percentile_signal(block=False)

        if self._percentiles is None:
            set_values(val_1=0, val_x=0, val_2=0, val_u=0, val_o=0)
            return

        odd_range = self._odd_ranges[self._combo_range.currentIndex()]

        if odd_range not in self._percentiles:
            set_values(val_1=0, val_x=0, val_2=0, val_u=0, val_o=0)
            return

        percentiles = self._percentiles[odd_range]
        set_values(
            val_1=percentiles['1'][0],
            val_x=percentiles['X'][0],
            val_2=percentiles['2'][0],
            val_u=percentiles['U'][0],
            val_o=percentiles['O'][0]
        )

    def _load_model(self):
        """ Load model, model_config, percentiles and restore its percentiles. """

        # Loading model, model_config.
        model_id = self._combo_model.currentText()
        self._model, self._model_config = self._model_db.load_model(model_id=model_id)

        # Loading percentiles.
        if not ('eval' in self._model_config and 'percentiles' in self._model_config['eval']):
            self._percentiles = None
        else:
            self._percentiles = self._model_config['eval']['percentiles']

        # Loading train-eval samples and updating dataset masks.
        eval_samples_size = self._model_config['train']['eval_samples_size']
        self._num_eval_samples = int(math.floor(self._df.shape[0]*eval_samples_size/100))
        self._num_train_samples = self._df.shape[0] - self._num_eval_samples
        self._dataset_mask_dict.update({
            'Train': np.array([False]*self._num_eval_samples + [True]*self._num_train_samples, dtype=bool),
            'Eval': np.array([True]*self._num_eval_samples + [False]*self._num_train_samples, dtype=bool),
        })

    def _predict(self):
        """ Predicts the targets, probabilities and updates correct mask. """

        # Generating predictions.
        y_prob = self._model.predict_proba(df=self._df)
        self._y_pred = y_prob.argmax(axis=1)
        self._y_prob = y_prob.round(2)

        # Computing correct mask.
        y_true = self._y_true_dict[self._target_types[self._combo_target.currentText()]]
        self._correct_mask = self._y_pred == y_true

    def _update_prob_percentiles(self):
        if self._y_prob is None:
            return

        # Filter probabilities by dataset (All/Eval/Train)
        dataset_mask = self._dataset_mask_dict[self._combo_dataset.currentText()]
        y_prob = self._y_prob[dataset_mask]

        # Filter probabilities by target.
        target_type = self._target_types[self._combo_target.currentText()]

        if target_type == TargetType.RESULT:
            p1 = self._slider_percentile_1.value()
            px = self._slider_percentile_X.value()
            p2 = self._slider_percentile_2.value()
            quantiles = np.quantile(y_prob, [p1/100, px/100, p2/100])
            self._prob_percentiles = {
                '1': (p1, quantiles[0]),
                'X': (px, quantiles[1]),
                '2': (p2, quantiles[2]),
                'U': (0, 0.0),
                'O': (0, 0.0)
            }
        elif target_type == TargetType.OVER_UNDER:
            pu = self._slider_percentile_under.value()
            po = self._slider_percentile_over.value()
            quantiles = np.quantile(y_prob, [pu/100, po/100])
            self._prob_percentiles = {
                '1': (0, 0.0),
                'X': (0, 0.0),
                '2': (0, 0.0),
                'U': (pu, quantiles[0]),
                'O': (po, quantiles[1])
            }
        else:
            raise ValueError(f'Undefined targets: "{target_type}"')

    def _update_table_predictions(self):
        """ Updates the table predictions/probabilities. """

        if self._y_prob is None:
            columns = ['Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)', 'Prob(U)', 'Prob(O)']
            data = np.array([['']*self._df.shape[0] for _ in range(6)], dtype=str).transpose()
            self._table.modify_columns(columns=columns, data=data)
            return

        if self._y_pred.ndim != 1:
            raise ValueError(f'Expected y_pred to be 1D, got {self._y_pred.shape}')
        if self._y_prob.ndim != 2:
            raise ValueError(f'Expected y_prob to be 2D, got {self._y_prob.shape}')

        # Convert predictions to labels.
        target_type = self._target_types[self._combo_target.currentText()]

        if target_type == TargetType.RESULT:
            mapper = np.array(['H', 'D', 'A'])
            columns = ['Predicted', 'Prob(1)', 'Prob(X)', 'Prob(2)']
        else:
            mapper = np.array(['U', 'O'])
            columns = ['Predicted', 'Prob(U)', 'Prob(O)']

        mapped_y_pred = mapper.take(self._y_pred)
        data = np.hstack([np.expand_dims(mapped_y_pred, axis=-1), self._y_prob])
        self._table.modify_columns(columns=columns, data=data)

    def _compute_hidden_mask(self):
        """ Computes hidden mask (matches/rows to hide). """

        # Filter by dataset.
        dataset_mask = self._dataset_mask_dict[self._combo_dataset.currentText()]

        # Filter by range.
        odd_range = self._odd_ranges[self._combo_range.currentIndex()]

        if odd_range != 'None':
            odd, low, high = odd_range
            odd_df = self._df[odd]
            mask = dataset_mask & ((low <= odd_df) & (odd_df <= high))
        else:
            mask = dataset_mask

        # Filter by percentile.
        if self._y_prob is None:
            return mask

        target_type = self._target_types[self._combo_target.currentText()]
        if target_type == TargetType.RESULT:
            thresholds = np.float32([self._prob_percentiles['1'][1], self._prob_percentiles['X'][1], self._prob_percentiles['2'][1]])
        else:
            thresholds = np.float32([self._prob_percentiles['U'][1], self._prob_percentiles['O'][1]])

        percentile_mask = np.all(self._y_prob >= thresholds, axis=1)
        final_mask = mask & percentile_mask
        return final_mask

    def _update_table_and_metrics(self):
        """ Updates table matches, highlights the correct ones and updates metrics. """

        def _update():
            # Set the percentile tooltips based on the selected percentile probabilities.
            self._set_percentile_tooltips()

            # Compute hidden mask matches.
            filter_mask = self._compute_hidden_mask()

            # Hide matches.
            hidden_row_ids = self._df[~filter_mask].index.tolist()
            self._table.set_new_hidden_rows(row_ids=hidden_row_ids)

            # Highlight matches and update metrics.
            if self._y_pred is not None:
                highlight_mask = self._correct_mask & filter_mask
                highlight_ids = self._df[highlight_mask].index.tolist()
                self._table.highlight_rows(row_ids=highlight_ids)

                target_type = self._target_types[self._combo_target.currentText()]
                y_true = self._y_true_dict[target_type][filter_mask]
                y_pred = self._y_pred[filter_mask]
                metrics_df = self._model.compute_metrics(y_true=y_true, y_pred=y_pred)
                self._acc_label.setText(f'Accuracy: {metrics_df.at[0, "Accuracy"]}')
                self._f1_label.setText(f'F1: {metrics_df.at[0, "F1"]}')
                self._prec_label.setText(f'Precision: {metrics_df.at[0, "Precision"]}')
                self._rec_label.setText(f'Recall: {metrics_df.at[0, "Recall"]}')
                self._samples_label.setText(f'Correct: {len(highlight_ids)}/{y_pred.shape[0]}')
                self._profit_balance_label.setText(f'Prof. Balance: {self._compute_profit_balance(y_pred=y_pred, filter_mask=filter_mask)}')
            else:
                self._acc_label.setText('Accuracy: 0.0')
                self._f1_label.setText('F1: 0.0')
                self._prec_label.setText('Precision: 0.0')
                self._rec_label.setText('Recall: 0.0')
                self._samples_label.setText(f'Samples: {self._df.shape[0]}')
                self._profit_balance_label.setText('Prof. Balance: 0.0')

        TaskRunnerDialog(title='Updating Table', info='Evaluating matches...', parent=self, task_fn=_update).run()

    def _compute_profit_balance(self, y_pred: pd.Series, filter_mask: np.ndarray) -> float:
        target_type = self._target_types[self._combo_target.currentText()]

        if target_type == TargetType.RESULT:
            odds_df = self._df.loc[filter_mask, ['1', 'X', '2']]
        else:
            return 0.0

        odds = odds_df.values[np.arange(y_pred.shape[0]), y_pred]
        profit_balance = compute_profit_balance(odds=odds)
        return profit_balance

    def _set_percentile_tooltips(self):
        target_type = self._target_types[self._combo_target.currentText()]

        if self._prob_percentiles is None:
            tooltip_1 = 'Prob(1): 0.0'
            tooltip_x = 'Prob(X): 0.0'
            tooltip_2 = 'Prob(2): 0.0'
            tooltip_u = 'Prob(U-2.5): 0.0'
            tooltip_o = 'Prob(O-2.5): 0.0'
        elif target_type == TargetType.RESULT:
            tooltip_1 = f'Prob(1): {round(self._prob_percentiles["1"][1], 2)}'
            tooltip_x = f'Prob(X): {round(self._prob_percentiles["X"][1], 2)}'
            tooltip_2 = f'Prob(2): {round(self._prob_percentiles["2"][1], 2)}'
            tooltip_u = 'Prob(U-2.5): 0.0'
            tooltip_o = 'Prob(O-2.5): 0.0'
        elif target_type == TargetType.OVER_UNDER:
            tooltip_1 = 'Prob(1): 0.0'
            tooltip_x = 'Prob(X): 0.0'
            tooltip_2 = 'Prob(2): 0.0'
            tooltip_u = f'Prob(U-2.5): {round(self._prob_percentiles["U"][1], 2)}'
            tooltip_o = f'Prob(O-2.5): {round(self._prob_percentiles["O"][1], 2)}'
        else:
            raise NotImplementedError(f'Not implemented target_type: "{target_type}"')

        self._slider_percentile_1.setToolTip(tooltip_1)
        self._slider_percentile_X.setToolTip(tooltip_x)
        self._slider_percentile_2.setToolTip(tooltip_2)
        self._slider_percentile_under.setToolTip(tooltip_u)
        self._slider_percentile_over.setToolTip(tooltip_o)

    def _show_instructions(self):
        QMessageBox.information(
            self,
            'Evaluation Instructions',
            'Select the target type, model and the dataset you wish to evaluate.'
            'You can also specify odd-range and probability percentile filters to utilize during predictions.'
            'The filters will not change the model outputs, but will evaluate the model on the selected matches.'
        )
