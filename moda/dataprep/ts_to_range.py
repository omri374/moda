import pandas as pd


def ts_to_range(ts, time_range='1H'):
    if not isinstance(ts.index, pd.core.indexes.datetimes.DatetimeIndex):
        print('Wrong index type. Expecting pd.core.indexes.datetimes.DateTimeIndex')
        return

    if 'category' in ts:
        range_grp = ts.groupby([pd.TimeGrouper(time_range), 'category']).agg('count')
        range_grp.columns.values[0] = 'value'
        range_grp = range_grp[['value']]
        range_grp.index.names = ['date', 'category']
    else:
        range_grp = ts.groupby(pd.TimeGrouper(time_range)).agg('count')

    return range_grp
