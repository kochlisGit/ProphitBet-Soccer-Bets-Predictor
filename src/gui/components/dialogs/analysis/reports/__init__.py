from src.gui.components.dialogs.analysis.reports.bivariate import BivariateReport
from src.gui.components.dialogs.analysis.reports.dataset import DatasetOverviewReport
from src.gui.components.dialogs.analysis.reports.importance import FeatureImportanceReport
from src.gui.components.dialogs.analysis.reports.rules import RulesReport
from src.gui.components.dialogs.analysis.reports.selection import FeatureSelectionReport
from src.gui.components.dialogs.analysis.reports.teams import TeamPerformanceReport
from src.gui.components.dialogs.analysis.reports.univariate import UnivariateReport

REPORT_WIDGETS = {
    'Dataset': DatasetOverviewReport,
    'Univariate': UnivariateReport,
    'Bivariate': BivariateReport,
    'Selection': FeatureSelectionReport,
    'Importance': FeatureImportanceReport,
    'Rules': RulesReport,
    'Team': TeamPerformanceReport,
}
