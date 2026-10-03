import io
import logging
from typing import Optional

import pandas as pd

from backend.app.config import setup_ml_path
from backend.app.services.model_artifact import deserialize_model

logger = logging.getLogger(__name__)


def generate_shap_plot(model_bytes: bytes, df: pd.DataFrame, plot_type: str = "bar") -> bytes:
    """
    Generate a SHAP explainability plot for the given model and data.
    Returns PNG image bytes.
    """
    setup_ml_path()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = deserialize_model(model_bytes)
    columns = getattr(model, "_input_columns", None)
    if "ScoreScope" in df.columns:
        df = df[df["ScoreScope"] == "regulation"]
    if columns:
        df = df[columns]

    from src.interpretability.explainer import ClassifierExplainer

    try:
        explainer = ClassifierExplainer(model=model, df=df)
        explainer.compute_shap_values()

        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "bar":
            ax = explainer.shap_bar_plot()
        elif plot_type == "partial_dependence":
            feature_idx = 0
            ax = explainer.partial_dependence_plot(feature_index=feature_idx)
        elif plot_type == "boundary":
            ax = explainer.boundary_plot()
        else:
            ax = explainer.shap_bar_plot()

        buf = io.BytesIO()
        fig = ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        logger.error(f"Explainability plot generation failed: {e}")
        plt.close("all")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Explainability not available:\n{str(e)[:100]}",
                ha="center", va="center", fontsize=12, color="gray",
                transform=ax.transAxes)
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.read()


EXPLAINABILITY_PLOT_TYPES = [
    {"value": "bar", "label": "SHAP Feature Importance (Bar)"},
    {"value": "partial_dependence", "label": "Partial Dependence"},
    {"value": "boundary", "label": "Decision Boundary"},
]
