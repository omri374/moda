import pandas as pd


def ts_to_range(ts, time_range='1H'):
    """
    Creates a new data frame with counts per time range
    :param ts: The original time series, with a column named 'date' and possibly an additional 'category' column
    :param time_range: The time range requested
    :return: a pd.DataFrame with a MultiIndex containing a date and category. Contains an additional column with
    counts in the interval
    """
    if not isinstance(ts.index, pd.core.indexes.datetimes.DatetimeIndex):
        print('Wrong index type. Expecting pd.core.indexes.datetimes.DateTimeIndex')
        return

    if 'category' in ts:
        ts = ts.loc[:,['date','category']] # Keep only first two columns
        range_grp = ts.groupby([pd.Grouper(freq=time_range), 'category']).agg('count')
        range_grp.columns.values[0] = 'value'
        range_grp = range_grp[['value']]
        range_grp.index.names = ['date', 'category']
    else:
        range_grp = ts.groupby(pd.Grouper(freq=time_range)).agg('count')

    return range_grp
