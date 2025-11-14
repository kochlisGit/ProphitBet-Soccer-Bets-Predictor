import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from matplotlib.axes import Axes
from src.analysis.analyzer import FeatureAnalyzer
from src.preprocessing.dataset import DatasetPreprocessor


class DescriptiveAnalyzer(FeatureAnalyzer):
    """ Descriptive Statistics analyzer for a provided league. """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df=df)

    def _generate_plot(self, df: pd.DataFrame, colormap: Optional[str] = None, feature_type: Optional[str] = None) -> Axes:
        """ Generates a descriptive statistics table for each variable. """

        # Calculate descriptive statistics.
        input_df = df[self._trainable_features + ['Result', 'Result-U/O']]

        # Select features.
        if feature_type is not None:
            if feature_type == 'home':
                input_df = input_df[['1', 'X', '2'] + [col for col in input_df.columns if col[0] == 'H']]
            elif feature_type == 'away':
                input_df = input_df[['1', 'X', '2'] + [col for col in input_df.columns if col[0] == 'A']]
            else:
                raise ValueError(f'Undefined feature type: "{feature_type}"')

        desc_table = input_df.describe().round(decimals=2)

        # Initialize figure.
        fig = plt.gcf() if plt.fignum_exists(1) else plt.figure()
        _, ax = plt.subplots(constrained_layout=True)
        ax.axis('off')
        fig.patch.set_facecolor('#1E90FF')

        # Generate table plot.
        tbl = ax.table(
            cellText=desc_table.values,
            rowLabels=desc_table.index,
            colLabels=desc_table.columns,
            cellLoc='center',
            loc='center'
        )

        # Style table.
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0 or col == -1:
                cell.get_text().set_fontweight('bold')

        # 5. title
        ax.set_title('Descriptive Statistics')
        return ax
