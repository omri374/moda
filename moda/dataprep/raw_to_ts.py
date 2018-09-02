import numpy as np
import pandas as pd


def raw_to_ts(raw):
    """
    Turns a raw pd.DataFrame into a time-series DataFrame, by creating a DatetimeIndex and a 'timestamp' column
    :param raw: a pd.DataFrame with a date column
    :return: a time-series DataFrame
    """
    if 'date' not in raw:
        raise ValueError("File must contain a date column")

    # raw = select_cols.rename(columns = columns)
    raw['date'] = pd.to_datetime(raw['date'])
    raw.set_index(pd.DatetimeIndex(raw['date']), inplace=True)
    raw['timestamp'] = raw.index.astype(np.int64) // 10 ** 9
    return raw



