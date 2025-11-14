import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from typing import Optional
from matplotlib.pyplot import Axes
from src.models.model import ClassificationModel
from src.interpretability.explainer import ClassifierExplainer


class NeuralNetworkExplainer(ClassifierExplainer):
    def __init__(self, model: ClassificationModel, df: pd.DataFrame):
        super().__init__(model=model, df=df)

        self._attn_model = model.classifier.attn_model
        self._supports_attention = self._attn_model is not None

    @property
    def supports_attention(self) -> bool:
        return self._supports_attention

    def _compute_shap_values(self) -> shap.Explanation:
        """ Returns shap values of Tree-based explainer. """

        explainer = shap.GradientExplainer(model=self._model.classifier.model, data=self._df)
        return explainer(self._df)

    def plot_attention_scores(self, colormap: Optional[str] = None) -> Optional[Axes]:
        """ Generates an attention-based bar plot that highlights the importance of each feature. """

        if not self._supports_attention:
            raise ValueError(
                'Attention model has not been built, as VSN is not activated. Cannot generate attention weights.'
            )

        scores = self._attn_model(self._x, training=False).numpy().mean(axis=0)

        scores_df = pd.DataFrame({'Features': self._df.columns.tolist(), 'Attention': scores})
        scores_df = scores_df.sort_values(by='Attention', ascending=False, ignore_index=True).iloc[: self._max_features]

        # Generate importance bar plot.
        _, ax = plt.subplots(constrained_layout=True)
        ax = sns.barplot(x=scores_df['Attention'], y=scores_df['Features'], palette=colormap, ax=ax)
        ax.set_ylabel(None)
        ax.tick_params(axis='both', labelsize='small')
        ax.set_title(f'Attention Bar Plot for {self._model.target_type.name}')
        return ax
