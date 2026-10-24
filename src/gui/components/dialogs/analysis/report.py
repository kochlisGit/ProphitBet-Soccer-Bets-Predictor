from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QFrame, QScrollArea, QVBoxLayout, QWidget
from src.gui.components.dialogs.analysis.reports import REPORT_WIDGETS

def _patch_report_items(report_dict: Dict[str, Any]) -> Dict[str, Any]:
    """ Add cross-analysis results required by the report widgets.

        Attributes
        ----------
        :param report_dict: (Dict[str, Any]) resulting report dict from analysis.
    """

    univariate_report = report_dict.get('Univariate', None)
    bivariate_report = report_dict.get('Bivariate', None)

    if 'Dataset' in report_dict:
        if univariate_report is not None:
            report_dict['Dataset']['imbalanced_variables'] = univariate_report['imbalanced_variables']
            report_dict['Dataset']['issues'] = univariate_report['issues']
        if bivariate_report is not None:
            report_dict['Dataset']['correlated_pairs'] = bivariate_report['correlated_pairs']
    if univariate_report is not None and bivariate_report is not None:
        report_dict['Univariate']['correlated_pairs'] = bivariate_report['correlated_pairs']

    return report_dict


class AnalysisReport(QDialog):
    """ Analysis Report Dialog that displays all selected analysis reports on one scrollable page. """

    def __init__(self, report_dict: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle('Data Analysis Report')
        self.setWindowIcon(QIcon('storage/icons/menu/ball.png'))
        self.setMinimumSize(900, 520)
        self.resize(900, 520)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setModal(True)

        report_dict = _patch_report_items(report_dict=report_dict)

        page = QWidget(self)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(12, 12, 12, 12)
        page_layout.setSpacing(16)

        for name, report_cls in REPORT_WIDGETS.items():
            report_items = report_dict.get(name)
            if not isinstance(report_items, dict):
                continue

            has_results = getattr(report_cls, 'has_results', None)
            if callable(has_results) and not has_results(**report_items):
                continue

            widget = report_cls(**report_items, parent=page)
            page_layout.addWidget(widget)
        page_layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(page)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
