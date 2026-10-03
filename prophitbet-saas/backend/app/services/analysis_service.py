import io
import logging
from typing import Optional

import pandas as pd

from backend.app.config import setup_ml_path

logger = logging.getLogger(__name__)

ANALYZER_MAP = {}


def _ensure_imports():
    global ANALYZER_MAP
    if ANALYZER_MAP:
        return
    setup_ml_path()
    from src.analysis import (
        CorrelationAnalyzer,
        DistributionAnalyzer,
        VarianceAnalyzer,
        BorutaAnalyzer,
        RuleExtractorAnalyzer,
        CoefficientAnalyzer,
        GiniImpurityAnalyzer,
        DescriptiveAnalyzer,
    )
    ANALYZER_MAP.update({
        "correlation": CorrelationAnalyzer,
        "distribution": DistributionAnalyzer,
        "variance": VarianceAnalyzer,
        "boruta": BorutaAnalyzer,
        "rules": RuleExtractorAnalyzer,
        "coefficients": CoefficientAnalyzer,
        "impurity": GiniImpurityAnalyzer,
        "description": DescriptiveAnalyzer,
    })


def run_analysis(
    df: pd.DataFrame,
    analysis_type: str,
    season: Optional[int] = None,
    colormap: str = "RdYlGn",
    **kwargs,
) -> bytes:
    """Run an analysis and return the resulting plot as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_imports()
    analyzer_cls = ANALYZER_MAP.get(analysis_type)
    if analyzer_cls is None:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    analyzer = analyzer_cls(df=df)
    ax = analyzer.generate_plot(season=season, colormap=colormap, **kwargs)

    buf = io.BytesIO()
    fig = ax.get_figure()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


TIER_ANALYSIS_ACCESS = {
    "free": {"description"},
    "pro": {
        "description", "correlation", "distribution", "variance",
        "boruta", "rules", "coefficients", "impurity",
    },
    "elite": {
        "description", "correlation", "distribution", "variance",
        "boruta", "rules", "coefficients", "impurity",
    },
}


def get_available_analyses(plan: str) -> list[str]:
    return sorted(TIER_ANALYSIS_ACCESS.get(plan, TIER_ANALYSIS_ACCESS["free"]))
