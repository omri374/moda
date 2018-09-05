import numpy as np
import pandas as pd


def raw_to_ts(raw, min_date=None, max_date=None):
    """
    Turns a raw pd.DataFrame into a time-series DataFrame, by creating a DatetimeIndex and a 'timestamp' column
    :param min_date: Minimum date for the time series
    :param max_date: Maximum date for the time series
    :param raw: a pd.DataFrame with a date column
    :return: a time-series DataFrame
    """
    if 'date' not in raw:
        raise ValueError("File must contain a date column")




    raw['date'] = pd.to_datetime(raw['date'])
    if min_date is not None:
        raw = raw[raw['date'] >= min_date]
    if max_date is not None:
        raw = raw[raw['date'] <= max_date]

    raw.set_index(pd.DatetimeIndex(raw['date']), inplace=True)
    raw.loc[:,'timestamp'] = raw.index.astype(np.int64) // 10 ** 9
    return raw
