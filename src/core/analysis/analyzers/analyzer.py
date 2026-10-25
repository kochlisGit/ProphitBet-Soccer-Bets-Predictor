import pandas as pd
from abc import ABC, abstractmethod
from typing import Any


class Analyzer(ABC):
    """ Base class for all analyzers. Each analyzer must return a dict (report) of the analysis. """

    def __init__(self, name: str):
        """
            Args
            ----
            :param name: (str): Name of the analyzer.
        """

        self._name = name
        self._decimals = 3

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def analyze(self, df: pd.DataFrame, metadata: pd.DataFrame) -> Any:
        """ Analyze the total dataset (all seasons), and each season separately.

            Args
            ----
            :param df: (pd.DataFrame): League's dataframe with computed statistics.
            :param metadata: (pd.DataFrame): League's metadata.
        """

        pass
