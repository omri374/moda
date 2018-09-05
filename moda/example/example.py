import pandas as pd

from moda.dataprep.raw_to_ts import raw_to_ts
from moda.dataprep.ts_to_range import ts_to_range


def raw_to_timeseries(datapath, time_range='30min', nrows=None, min_date=None, max_date=None, save_files=False):
    """
    Take a raw data with timestamps (date column), categories (category column) and additional columns,
    and turn it into a ranged time-series: Group the original raw data by time interval (time_range) and category.
    Result is the number of samples per category in each time range.
    :param datapath: the path to the csv file
    :param time_range: the time_range according to which the data is grouped by
    :param nrows: limits the number of rows read from the csv
    :param min_date: filters out ranges prior to min_date
    :param max_date: filters out ranges after max_date
    :param save_files: Whether to save intermediate csvs
    :returns a pd.DataFrame with one value per time_range and category.
    This value is the number of samples within this range for a specific category


    """
    if nrows is None:
        raw = pd.read_csv(datapath, usecols=['Opened', 'Category'])
    else:
        raw = pd.read_csv(datapath, usecols=['Opened', 'Category'], nrows=nrows)

    raw = raw.rename(columns={'Opened': 'date', 'Category': 'category'})

    # Create a time series dataframe
    ts = raw_to_ts(raw, min_date=min_date, max_date=max_date)

    # Divide time series to ranges and categories
    ranged_ts = ts_to_range(ts, time_range=time_range)

    if save_files:
        if nrows is None:
            ts.to_csv("ts_SF311.csv")
            ranged_ts.to_csv("ranged_ts_SF311.csv")
        else:
            ts.to_csv("ts_SF311_" + str(nrows) + "_rows.csv")
            ranged_ts.to_csv("ranged_ts_SF311_" + str(nrows) + "_rows.csv")
    return ranged_ts

if __name__ == '__main__':
    range = raw_to_timeseries('SF311_simplified.csv')
