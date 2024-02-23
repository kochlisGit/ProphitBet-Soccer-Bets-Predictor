import pandas as pd
from abc import ABC, abstractmethod


class FeatureAnalyzer(ABC):
    def __init__(self, df: pd.DataFrame, preprocess: bool = True):
        input_df = df.dropna()

        if preprocess:
            self._input_df = input_df.drop(columns=['Date', 'Season', 'Home Team', 'Away Team']).reset_index(drop=True)
        else:
            self._input_df = input_df

        self._columns = self._input_df.columns.tolist()
        self._colormap = {
            'Blues': 'Blues',
            'Coolwarm': 'coolwarm',
            'Crest': 'crest',
            'Rocket': 'rocket',
            'Icefire': 'icefire'
        }

    @property
    def input_df(self) -> pd.DataFrame:
        return self._input_df

    @property
    def columns(self) -> list:
        return self._columns

    @property
    def colormap(self) -> dict[str, str]:
        return self._colormap

    @abstractmethod
    def plot(self, ax, **kwargs):
        pass
