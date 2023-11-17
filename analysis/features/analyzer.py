import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum


class FeatureAnalyzer(ABC):
    class ColorMaps(Enum):
        Coolwarm = "coolwarm"
        Rocket = "rocket"
        Icefire = "icefire"
        Crest = "crest"
        Blues = "Blues"

    def __init__(self, matches_df: pd.DataFrame):
        self._inputs = matches_df.drop(
            columns=["Season", "Date", "Home Team", "Away Team", "Result"]
        ).astype(np.float32)
        self._targets = matches_df["Result"].replace({"H": 0, "D": 1, "A": 2})
        self._columns = self._inputs.columns

    @property
    def inputs(self) -> pd.DataFrame:
        return self._inputs

    @property
    def targets(self) -> pd.DataFrame:
        return self._targets

    @property
    def columns(self) -> list:
        return self._columns

    @abstractmethod
    def plot(
        self,
        x: np.ndarray or list,
        y: np.ndarray or list,
        color_map: str,
        mask: np.ndarray,
        ax,
    ):
        pass
