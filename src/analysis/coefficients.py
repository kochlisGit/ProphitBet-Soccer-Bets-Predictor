import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Optional, Tuple
from matplotlib.axes import Axes
from sklearn.linear_model import LogisticRegressionCV
from src.analysis.analyzer import FeatureAnalyzer
from src.preprocessing.dataset import DatasetPreprocessor
from src.preprocessing.utils.normalization import NormalizerType
from src.preprocessing.utils.target import TargetType


class CoefficientAnalyzer(FeatureAnalyzer):
    """ Logistic Regression Coefficient analyzer.
        The absolute value of each linear coefficient shows the importance of the corresponding feature.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

        self._preprocessor = DatasetPreprocessor()
        self._max_features = 20

    def _generate_plot(
            self,
            df: pd.DataFrame,
            colormap: Optional[str] = None,
            target_type: TargetType = None
    ) -> Axes:
        """
            Applies Standard scaling to data and Generates a coefficient importance
            bar plot for the specified classification task.
        """

        # Compute linear coefficients.
        coef, classes = self._compute_linear_coefficients(df=df, target_type=target_type)
        n_classes = len(classes)

        # Generate bar plots for binary classification tasks.
        if n_classes < 3:
            coef_df = pd.DataFrame({'Features': self._trainable_features, 'Coefficients': coef[0]})
            ax = self._generate_coefficient_plot(coef_df=coef_df, vertical_labels=True, ax=None, colormap=colormap)
            ax.set_title(f'Top-{self._max_features} Feature Importance Bar Plot for {classes[0]}-{classes[1]}')
        else:
            fig, axes = plt.subplots(nrows=n_classes, ncols=1)
            fig.suptitle(f'Top-{self._max_features} Feature Importance Bar Plot for {classes}')
            for idx, cls_label in enumerate(classes):
                ax = axes[idx]
                coef_df = pd.DataFrame({'Features': self._trainable_features, 'Coefficients': coef[idx]})
                _ = self._generate_coefficient_plot(coef_df=coef_df, vertical_labels=False, ax=ax, colormap=colormap)
                ax.set_ylabel(f'{cls_label} Coefficients')
            ax = axes
        return ax

    def _compute_linear_coefficients(self, df: pd.DataFrame, target_type: TargetType) -> Tuple[np.ndarray, List[str]]:
        """ Fits a Logistic Regression model in the dataset and returns the linear coefficients, classes. """

        # Construct & Normalize inputs.
        x, y, _ = self._preprocessor.preprocess_dataset(
            df=df,
            target_type=target_type,
            normalizer=NormalizerType.STANDARD,
            sampler=None
        )

        # Apply Linear Regression with Lasso (L1 Penalty).
        model = LogisticRegressionCV(
            Cs=[1e-3, 1e-2, 1e-1, 1.0, 2],
            penalty='l1',
            solver='saga',
            max_iter=5000,
            random_state=0,
            n_jobs=-1
        )
        model.fit(x, y)

        # Get classes, coefficients
        if target_type == TargetType.RESULT:
            classes = ['H', 'D', 'A']
        elif target_type == TargetType.OVER_UNDER:
            classes = ['U', 'O']
        else:
            raise ValueError(f'Not defined target type: {target_type.name}')

        return np.abs(model.coef_), classes

    def _generate_coefficient_plot(
            self,
            coef_df: pd.DataFrame,
            vertical_labels: bool = True,
            ax: Optional[Axes] = None,
            colormap: Optional[str] = None
    ) -> Axes:
        """ Generates the coefficient bar plot of the Top-K features. """

        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        if vertical_labels:
            x_label = 'Coefficients'
            y_label = 'Features'
        else:
            x_label = 'Features'
            y_label = 'Coefficients'

        coef_df = coef_df.sort_values(by='Coefficients', ascending=False, ignore_index=True).iloc[: self._max_features]
        ax = sns.barplot(
            x=coef_df[x_label],
            y=coef_df[y_label],
            ax=ax,
            palette=colormap
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.tick_params(axis='both', labelsize='small')
        return ax
