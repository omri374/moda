import pandas as pd


def create_windowed_df(ranged_ts, window_size, with_actual=True):
    """
    Creates a data frame where each row is a window of samples from the time series.
    Each consecutive row is a shift of 1 cell from the previous row.
    For example: [[1,2,3],[2,3,4],[3,4,5]]

    :param ranged_ts: a pd.DataFrame containing one column for values and one pd.DatetimeIndex for dates
    :param window_size: The number of timestamps to be used as features
    :param with_actual: Whether to increase window size by one, and treat the last column as the ground truth
    (relevant for forecasting scenarios). Returns the same output just with a window size bigger by 1.
    :return:
    """
    ranged_ts_copy = ranged_ts.copy()
    for i in range(window_size - 1 + int(with_actual)):
        ranged_ts = pd.concat([ranged_ts, ranged_ts_copy.shift(-(i + 1))], axis=1)
        ranged_ts = ranged_ts.dropna(axis=0)

    return ranged_ts


if __name__ == '__main__':
    ranged_ts = pd.DataFrame({'date': range(6), 'value': range(6)})
    ranged_ts['date'] = pd.to_datetime(ranged_ts['date'])
    ranged_ts = ranged_ts.set_index(pd.DatetimeIndex(ranged_ts['date']))
    ranged_ts = ranged_ts.drop(columns='date')
    ranged_ts.head()
    windowed_df = create_windowed_df(ranged_ts, window_size=3, with_actual=False)
