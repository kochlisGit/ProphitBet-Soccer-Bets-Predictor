"""Reject suspicious result datasets before replacing a usable snapshot."""
import pandas as pd


def validate_dataset(df, previous_rows=0):
    required = {'Date', 'Home', 'Away', 'HG', 'AG', 'Result'}
    if df.empty or not required.issubset(df.columns):
        raise ValueError('Dataset is empty or missing result columns')
    if previous_rows and len(df) < previous_rows * .8:
        raise ValueError('Dataset row count dropped by more than 20%; previous snapshot preserved')
    dates = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    if dates.isna().any() or (dates > pd.Timestamp.now(tz='UTC')).any():
        raise ValueError('Result dataset contains invalid or future dates')
    home = df['Home'].astype('string').str.strip().str.casefold()
    away = df['Away'].astype('string').str.strip().str.casefold()
    if home.isna().any() or away.isna().any() or (home == '').any() or (away == '').any() or (home == away).any():
        raise ValueError('Dataset contains invalid team identities')
    hg, ag = pd.to_numeric(df['HG'], errors='coerce'), pd.to_numeric(df['AG'], errors='coerce')
    if any(((s.isna()) | (s < 0) | (s % 1 != 0)).any() for s in (hg, ag)):
        raise ValueError('Dataset contains invalid final scores')
    expected = pd.Series('D', index=df.index).mask(hg > ag, 'H').mask(hg < ag, 'A')
    if not df['Result'].eq(expected).all():
        raise ValueError('Dataset result labels contradict final scores')
    if pd.DataFrame({'date': dates, 'home': home, 'away': away}).duplicated().any():
        raise ValueError('Dataset contains duplicate match identities')
