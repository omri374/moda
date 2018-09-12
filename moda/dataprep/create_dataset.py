import pandas as pd


def get_windowed_ts(ranged_ts, window_size, with_actual=True):
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

    windowed_ts = ranged_ts

    windowed_ts_copy = windowed_ts.copy()
    for i in range(window_size - 1 + int(with_actual)):
        windowed_ts = pd.concat([windowed_ts, windowed_ts_copy.shift(-(i + 1))], axis=1)
        windowed_ts = windowed_ts.dropna(axis=0)

    return windowed_ts


def split_history_and_current(windowed_ts):
    """
    Returns the first n-1 columns as X, and the last column as y. Useful mainly for forecasting scenarios
    :param windowed_ts: a pd.DataFrame with a date index and a column per timestamp. see get_windowed_ts
    :return:
    """
    X = windowed_ts.iloc[:, :-1].values
    y = windowed_ts.iloc[:, -1].values

    return (X, y)


if __name__ == '__main__':
    ranged_ts = pd.DataFrame({'date': range(6), 'value': range(6)})
    ranged_ts['date'] = pd.to_datetime(ranged_ts['date'])
    ranged_ts = ranged_ts.set_index(pd.DatetimeIndex(ranged_ts['date']))
    ranged_ts = ranged_ts.drop(columns='date')
    ranged_ts.head()
    windowed_df = get_windowed_ts(ranged_ts, window_size=3, with_actual=False)
