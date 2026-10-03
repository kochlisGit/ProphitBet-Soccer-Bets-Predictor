"""Resolve repeated source rows before calculating any historical features."""
import logging
import pandas as pd


def deduplicate_results(df):
    """Merge identical match facts; reject contradictions and blank ambiguous odds."""
    result = df.copy().reset_index(drop=True)
    keys = pd.DataFrame({
        'date': pd.to_datetime(result['Date'], errors='raise', utc=True).dt.normalize(),
        'home': result['Home'].astype('string').str.strip().str.casefold(),
        'away': result['Away'].astype('string').str.strip().str.casefold(),
    })
    duplicates = keys.duplicated(keep=False)
    removed = []
    for _, group in keys[duplicates].groupby(['date', 'home', 'away'], dropna=False):
        indices = group.index
        rows = result.loc[indices]
        for column in result.columns:
            if column in ('Date', 'Home', 'Away'):
                continue
            if rows[column].nunique(dropna=False) > 1:
                if column in ('1', 'X', '2'):
                    result.loc[indices[0], column] = float('nan')
                else:
                    raise ValueError(f'Conflicting duplicate match facts in {column}; dataset rejected')
        removed.extend(indices[1:])
    if removed:
        logging.getLogger(__name__).warning('Removed %s repeated matches before feature calculation', len(removed))
    return result.drop(index=removed).reset_index(drop=True)
