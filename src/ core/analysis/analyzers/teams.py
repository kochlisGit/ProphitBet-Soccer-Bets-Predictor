import pandas as pd
from typing import Dict
from src.core.analysis.analyzers.analyzer import Analyzer


class TeamPerformanceAnalyzer(Analyzer):
    """ Team performance analyzer class. It analyzes the performance of each team per time.

        Performance:
            1. Total Wins per Time
            2. Total Goals per Time
            3.
    """

    def __init__(self):
        super().__init__(name='Team')

    def analyze(self, df: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        expected = df[['1', 'X', '2']].idxmin(axis=1).map({'1': 'H', 'X': 'D', '2': 'A'})
        reversed_result = metadata['Result'].map({'H': 'A', 'D': 'D', 'A': 'H'})
        reversed_expected = expected.map({'H': 'A', 'D': 'D', 'A': 'H'})
        fixture_order = range(len(metadata))

        home_performance = pd.DataFrame({
            'Date': metadata['Date'],
            'Season': metadata['Season'],
            'Week': metadata['Week'],
            'Team': metadata['Home'],
            'Goals': metadata['HG'],
            'Goals Against': metadata['AG'],
            'Result': metadata['Result'],
            'Expected Result': expected,
            '_Fixture Order': fixture_order,
            '_Side Order': 0,
        })
        away_performance = pd.DataFrame({
            'Date': metadata['Date'],
            'Season': metadata['Season'],
            'Week': metadata['Week'],
            'Team': metadata['Away'],
            'Goals': metadata['AG'],
            'Goals Against': metadata['HG'],
            'Result': reversed_result,
            'Expected Result': reversed_expected,
            '_Fixture Order': fixture_order,
            '_Side Order': 1,
        })
        performance = pd.concat([home_performance, away_performance], ignore_index=True)
        performance['Wins'] = performance['Result'].eq('H').astype(int)
        performance['Points'] = performance['Result'].map({'H': 3, 'D': 1, 'A': 0})

        performance = performance.sort_values(
            by=['Season', 'Date', '_Fixture Order', '_Side Order'],
            kind='stable',
            ignore_index=True,
        )
        grouped = performance.groupby(by=['Season', 'Team'], sort=False)

        # Computing timeline dataframe for each season, team pair.
        timeline_df = performance[['Date', 'Season', 'Week', 'Team']].copy()
        timeline_df[['Wins', 'Goals', 'Points']] = grouped[['Wins', 'Goals', 'Points']].cumsum()

        # Computing Seasonal Timeline Wins, Goals, Points, GF, GA.
        seasonal_timeline_df = grouped.agg(
            Wins=('Wins', 'sum'),
            Goals=('Goals', 'sum'),
            Points=('Points', 'sum'),
            GF=('Goals', 'mean'),
            AG=('Goals Against', 'mean'),
        ).reset_index()

        # Computing actual vs expected results
        actual_expected_df = performance.assign(
            **{
                'Total Wins': performance['Result'].eq('H'),
                'Total Expected Wins': performance['Expected Result'].eq('H'),
                'Total Draws': performance['Result'].eq('D'),
                'Total Expected Draws': performance['Expected Result'].eq('D'),
                'Total Losses': performance['Result'].eq('A'),
                'Total Expected Losses': performance['Expected Result'].eq('A'),
            }
        ).groupby('Team', sort=False)[[
            'Total Wins',
            'Total Expected Wins',
            'Total Draws',
            'Total Expected Draws',
            'Total Losses',
            'Total Expected Losses',
        ]].sum().astype(int).reset_index()
        return {
            'timeline': timeline_df,
            'seasonal_timeline': seasonal_timeline_df,
            'actual_expected_results': actual_expected_df
        }
