import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from matplotlib.axes import Axes
from src.preprocessing.dataset import DatasetPreprocessor


class FeatureAnalyzer(ABC):
    """ Base analyzer class. Each analyzer should generate a plot, which will be displayed in the GUI. """

    def __init__(self, df: pd.DataFrame):
        self._df = df

        if self._df.shape[0] == 0:
            raise ValueError('Nan-free dataframe has 0 samples.')

        seasons = df['Season']
        self._seasons = [None] + [i for i in range(seasons.iloc[-1], seasons.iloc[0] + 1)]
        self._trainable_features = df.columns.drop(DatasetPreprocessor().non_trainable_columns, errors='ignore').tolist()

        self._colormap_dict = {
            'Blues': 'Blues',
            'Coolwarm': 'coolwarm',
            'Crest': 'crest',
            'HUSL': 'husl',
            'Icefire': 'icefire',
            'Rocket': 'rocket',
            'Summer': 'summer'
        }

    @property
    def seasons(self) -> List[int]:
        return self._seasons

    @property
    def colormap_dict(self) -> Dict[str, str]:
        return self._colormap_dict

    def generate_plot(self, season: Optional[int] = None, colormap: Optional[str] = None, **kwargs) -> Axes:
        """ Analysis and generates a visualization plot.
            :param season: Season filter. If None, all seasons will be used in the analysis.
            :param colormap: Colormap filter. If None, the default colormap is used.
            :return: The generated plot on the provided ax.
        """

        # Filter by season.
        df = self._df
        filtered_df = df[df['Season'] == season] if season is not None else df

        if filtered_df.shape[0] == 0:
            available_seasons = sorted(df['Season'].unique().tolist())
            raise ValueError(f'Season: {season} is not found in dataframe. Available seasons are: {available_seasons}')

        # Analyze features and generate analysis plot.
        ax = self._generate_plot(df=filtered_df, colormap=colormap, **kwargs)
        return ax

    @abstractmethod
    def _generate_plot(self, df: pd.DataFrame, colormap: Optional[str], **kwargs) -> Axes:
        """ Analysis and generates a plot based on the provided ax and dataframe. """

        pass
