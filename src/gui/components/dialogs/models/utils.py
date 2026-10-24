import numpy as np
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QCheckBox, QDialog, QVBoxLayout, QWidget
from src.core.agents.utils import BETTING_PERFORMANCE_METRICS, compute_betting_metrics
from src.core.metrics.utils import (
    DOUBLE_CHANCE_ODD_COLUMNS,
    ODDS_COLUMNS,
    generate_report,
    generate_report_dc
)
from src.core.preprocessing.utils.targets import TargetType
from src.gui.widgets import ExcelTable

EXTRA_METRIC_COLUMNS = [
    'F1',
    'Precision',
    'Recall',
    'Log Loss',
    'Log Loss Skill',
    'Brier Score Skill',
    'RPS Skill',
    'MAPD'
]


def _compute_yearly_metrics(
        df: pd.DataFrame,
        seasons: List,
        metric_columns: List[str],
        selected_ids: pd.Index,
        current_metrics_df: pd.DataFrame,
        target_type: Optional[TargetType],
        y_pred_ids: Optional[np.ndarray],
        y_prob: Optional[np.ndarray],
        actions: Optional[np.ndarray],
        double_chance: bool = False
) -> pd.DataFrame:
    """Compute the current evaluation metrics for every season."""

    updated_metrics_df = pd.DataFrame(
        [[season] + [np.nan] * len(metric_columns) for season in ['All', *seasons]],
        columns=['Season', *metric_columns],
        dtype=object
    )
    updated_metrics_df.loc[0, metric_columns] = current_metrics_df.loc[0, metric_columns].to_numpy()

    selected_ids = df.index.intersection(selected_ids, sort=False)
    with tqdm(seasons, desc='Yearly Evaluation', unit='season') as progress_bar:
        for row_id, season in enumerate(progress_bar, start=1):
            progress_bar.set_description(f'Evaluating Season {season}')

            if QThread.currentThread().isInterruptionRequested():
                return updated_metrics_df

            season_ids = selected_ids[
                df.loc[selected_ids, 'Season'].to_numpy() == season
            ]
            if season_ids.empty:
                continue

            if target_type is not None and y_pred_ids is not None and y_prob is not None:
                if double_chance:
                    model_metrics_df = generate_report_dc(
                        y_true=df.loc[season_ids, TargetType.RESULT.value].to_numpy(),
                        y_pred=y_pred_ids[season_ids],
                        y_prob=y_prob[season_ids],
                        odds=df.loc[season_ids, DOUBLE_CHANCE_ODD_COLUMNS]
                    )
                    model_metrics_df.rename(columns={
                        f'MIP-{double_chance_odd}': f'MIP-{odd}'
                        for odd, double_chance_odd in zip(ODDS_COLUMNS, DOUBLE_CHANCE_ODD_COLUMNS)
                    }, inplace=True)
                else:
                    mapper = target_type.get_mapper()
                    y_true_labels = df.loc[season_ids, target_type.value]
                    model_metrics_df = generate_report(
                        y_true=y_true_labels.map(mapper).to_numpy(),
                        y_pred=y_pred_ids[season_ids],
                        y_prob=y_prob[season_ids],
                        odds=df.loc[season_ids, ODDS_COLUMNS],
                        target_type=target_type
                    )
                model_metric_columns = pd.Index(metric_columns).intersection(model_metrics_df.columns, sort=False)
                updated_metrics_df.loc[row_id, model_metric_columns] = (
                    model_metrics_df.loc[0, model_metric_columns].to_numpy()
                )

            if actions is not None:
                betting_metrics = compute_betting_metrics(
                    df=df.loc[season_ids],
                    actions=actions[season_ids]
                )
                updated_metrics_df.loc[row_id, BETTING_PERFORMANCE_METRICS] = list(betting_metrics)

    return updated_metrics_df


class _YearlyMetricsWorker(QObject):
    resultReady = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal()

    def __init__(self, request_id: int, request: dict):
        super().__init__()
        self._request_id = request_id
        self._request = request

    @Slot()
    def run(self):
        try:
            metrics_df = _compute_yearly_metrics(**self._request)
            self.resultReady.emit(self._request_id, metrics_df)
        except Exception as exception:
            self.failed.emit(self._request_id, str(exception))
        finally:
            self.finished.emit()


class YearlyEvaluationDialog(QDialog):
    """Non-modal evaluation report split by season."""

    metricsUpdated = Signal()

    def __init__(
            self,
            df: pd.DataFrame,
            metric_columns: List[str],
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._df = df
        self._seasons = self._df['Season'].drop_duplicates().tolist()
        self._metric_columns = metric_columns
        self._metrics_df = pd.DataFrame(
            [[season] + [np.nan] * len(metric_columns) for season in ['All', *self._seasons]],
            columns=['Season', *metric_columns],
            dtype=object
        )
        self._request_id = 0
        self._pending_request = None
        self._thread = None
        self._worker = None

        self.setWindowTitle('Yearly Evaluation Metrics')
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setMinimumSize(600, 400)
        self.resize(600, 400)

        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(8, 8, 8, 8)
        dialog_layout.setSpacing(6)

        self._metrics_table = ExcelTable(
            parent=self,
            df=self._metrics_df,
            readonly=True,
            supports_sorting=False,
            supports_query_search=False,
            supports_deletion=False,
            auto_resize=True
        )
        self._metrics_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._metrics_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._metrics_table.verticalHeader().hide()
        dialog_layout.addWidget(self._metrics_table)

        self._hide_extra_metrics_checkbox = QCheckBox('Hide Extra Metrics', self)
        self._hide_extra_metrics_checkbox.setChecked(False)
        self._hide_extra_metrics_checkbox.setToolTip('Hide extra evaluation metrics from the table.')
        self._hide_extra_metrics_checkbox.toggled.connect(self._hide_extra_metrics_changed)
        dialog_layout.addWidget(
            self._hide_extra_metrics_checkbox,
            alignment=Qt.AlignmentFlag.AlignLeft
        )

    def _hide_extra_metrics_changed(self, checked: bool):
        self._metrics_table.hide_columns(columns=EXTRA_METRIC_COLUMNS, hide=checked)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def update_metrics(
            self,
            selected_ids: pd.Index,
            current_metrics_df: pd.DataFrame,
            target_type: Optional[TargetType],
            y_pred_ids: Optional[np.ndarray],
            y_prob: Optional[np.ndarray],
            actions: Optional[np.ndarray],
            double_chance: bool = False
    ):
        """Queue the latest per-season metric calculation on a worker thread."""

        if not self.isVisible():
            return

        metric_labels = {
            f'MIP-{odd}': f'MIP-{double_chance_odd}'
            for odd, double_chance_odd in zip(ODDS_COLUMNS, DOUBLE_CHANCE_ODD_COLUMNS)
        } if double_chance else {}
        self._metrics_table.set_column_labels(metric_labels)

        self._request_id += 1
        self._pending_request = {
            'df': self._df,
            'seasons': self._seasons,
            'metric_columns': self._metric_columns,
            'selected_ids': selected_ids.copy(),
            'current_metrics_df': current_metrics_df.copy(deep=True),
            'target_type': target_type,
            'y_pred_ids': y_pred_ids,
            'y_prob': y_prob,
            'actions': actions,
            'double_chance': double_chance
        }
        self._start_pending_update()

    def shutdown(self):
        """Cancel queued work and finish the active worker before deletion."""

        self._request_id += 1
        self._pending_request = None

        thread = self._thread
        if thread is None:
            return

        thread.requestInterruption()
        thread.quit()
        thread.wait()
        self._thread = None
        self._worker = None
        thread.deleteLater()

    def _start_pending_update(self):
        if self._thread is not None or self._pending_request is None:
            return

        request_id = self._request_id
        request = self._pending_request
        self._pending_request = None

        self._thread = QThread(self)
        self._worker = _YearlyMetricsWorker(request_id=request_id, request=request)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.resultReady.connect(self._apply_metrics)
        self._worker.failed.connect(self._metrics_update_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()

    @Slot(int, object)
    def _apply_metrics(self, request_id: int, metrics_df: pd.DataFrame):
        if request_id != self._request_id:
            return

        self._metrics_df = metrics_df
        self._metrics_table.modify_rows(
            row_ids=list(range(self._metrics_df.shape[0])),
            items=self._metrics_df.values.tolist()
        )
        self.metricsUpdated.emit()

    @Slot(int, str)
    def _metrics_update_failed(self, request_id: int, error: str):
        if request_id == self._request_id:
            self._metrics_table.show_status(f'Could not compute yearly metrics: {error}', duration=5000)

    @Slot()
    def _thread_finished(self):
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()
        self._start_pending_update()
