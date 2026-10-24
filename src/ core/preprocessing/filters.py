import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass(slots=True)
class EvalFilter:
    """ Evaluation filter that can be formed in the evaluation mode.

        Attributes
        ----------
        :param prediction_filter: (int | None) predicted target index to retain.
        :param odd_filter: (str, float, float): tuple specifying the minimum and maximum odd value for a specified odd.
                           Supported odds are 1/X/2/1X/12/X2.
        :param percentile_filter: (str, float, float): tuple specifying the minimum probability and the percentile for
               a specified results. Supported results are H/D/A/U/O.
        :param double_chance: whether the filter was created for double-chance Result predictions.
    """

    prediction_filter: Optional[int] = None
    odd_filter: Optional[Tuple[str, float, float]] = None
    percentile_filter: Optional[Tuple[str, float, float]] = None
    double_chance: bool = False

    def __setstate__(self, state) -> None:
        """Restore filters saved before ``double_chance`` was introduced."""

        slot_values = state[1] if isinstance(state, tuple) else state
        self.prediction_filter = slot_values.get('prediction_filter')
        self.odd_filter = slot_values.get('odd_filter')
        self.percentile_filter = slot_values.get('percentile_filter')
        self.double_chance = slot_values.get('double_chance', False)


def filter_league_odds(
        df: pd.DataFrame,
        odd_1_mask_range: Optional[Tuple[float, float]] = None,
        odd_x_mask_range: Optional[Tuple[float, float]] = None,
        odd_2_mask_range: Optional[Tuple[float, float]] = None,
        validate_dataframe: bool = True
) -> Optional[pd.DataFrame]:
    """ Filters matches based on the provided odd ranges. Optionally, it validates whether the filtered dataframe
        is valid for EDA & Training.

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe.
        :param odd_1_mask_range: ((float, float) | None) tuple specifying the minimum and maximum odd 1 value.
        :param odd_x_mask_range: ((float, float) | None) tuple specifying the minimum and maximum odd X value.
        :param odd_2_mask_range: ((float, float) | None) tuple specifying the minimum and maximum odd 2 value.
        :param validate_dataframe: (bool) whether to validate the filtered dataframe.
    """

    def filter_odds(data_df: pd.DataFrame, odd: str, odd_filter: Optional[Tuple[float, float]]) -> pd.DataFrame:
        """ Helper function that filters matches for the specified odd.

            Args
            ----
            :param data_df: (pd.DataFrame) league's dataframe.
            :param odd: (str) specified odd (1/X/2).
            :param odd_filter: ((float, float) | None) tuple specifying the minimum and maximum odd value.
        """

        if odd_filter is None:
            return data_df

        min_odd, max_odd = odd_filter
        mask = ((data_df[odd] >= min_odd) & (data_df[odd] <= max_odd))
        return data_df[mask]

    # Applying odd filters.
    filtered_df = filter_odds(data_df=df, odd='1', odd_filter=odd_1_mask_range)
    filtered_df = filter_odds(data_df=filtered_df, odd='X', odd_filter=odd_x_mask_range)
    filtered_df = filter_odds(data_df=filtered_df, odd='2', odd_filter=odd_2_mask_range)

    # Validating processed dataframe.
    if validate_dataframe:
        valid_df = filtered_df.dropna()

        if valid_df.shape[0] == 0 or valid_df['Result'].nunique() < 3 or valid_df['Result-U/O'].nunique() < 2:
            return None

    return filtered_df.reset_index(drop=True)


def filter_matches(matches_df: pd.DataFrame, eval_filters: List[EvalFilter]) -> pd.DataFrame:
    """ Helper function that filters matches using evaluation filters that can be created in the evaluation mode.

        Args
        ----
        :param matches_df: (pd.DataFrame) league's dataframe.
        :param eval_filters: (List[EvalFilter]) list of evaluation filters.
    """

    def filter_prediction(prediction_filter: int) -> pd.Series:
        """Helper function that filters matches for the specified prediction index."""

        return matches_df['Prediction'] == prediction_filter

    def filter_odds(odd_filter: Tuple[str, float, float]) -> pd.Series:
        """ Helper function that filters matches for the specified odd. """

        odd, min_val, max_val = odd_filter
        return (matches_df[odd] >= min_val) & (matches_df[odd] <= max_val)

    def filter_percentiles(percentile_filter: Tuple[str, float, float]) -> pd.Series:
        """ Helper function that filters matches for the specified percentile. """

        result, min_prob, _ = percentile_filter
        return matches_df[f'Prob-{result}'] >= min_prob

    # Return if no filters are available.
    if not eval_filters:
        return matches_df

    mask = pd.Series(True, index=matches_df.index)
    for eval_filter in eval_filters:
        if eval_filter.prediction_filter is not None:
            mask &= filter_prediction(prediction_filter=eval_filter.prediction_filter)

        if eval_filter.odd_filter is not None:
            mask &= filter_odds(odd_filter=eval_filter.odd_filter)

        if eval_filter.percentile_filter is not None:
            mask &= filter_percentiles(percentile_filter=eval_filter.percentile_filter)
    return matches_df[mask]
