from src.core.analysis.analyzers.bivariate import BivariateAnalyzer
from src.core.analysis.analyzers.dataset import DatasetAnalyzer
from src.core.analysis.analyzers.importance import FeatureImportanceAnalyzer
from src.core.analysis.analyzers.rules import RuleExtractionAnalyzer
from src.core.analysis.analyzers.selection import FeatureSelectionAnalyzer
from src.core.analysis.analyzers.univariate import UnivariateAnalyzer
from src.core.analysis.analyzers.teams import TeamPerformanceAnalyzer

analyzers = [
    DatasetAnalyzer(),
    UnivariateAnalyzer(),
    BivariateAnalyzer(),
    FeatureSelectionAnalyzer(),
    FeatureImportanceAnalyzer(),
    RuleExtractionAnalyzer(),
    TeamPerformanceAnalyzer()
]
