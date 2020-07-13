import numpy as np
import pandas as pd


def raw_to_ts(raw, min_date=None, max_date=None, date_format=None):
    """
    Turns a raw pd.DataFrame into a time-series DataFrame, by creating a DatetimeIndex and a 'timestamp' column
    :param raw: a pd.DataFrame with a date column
    :param min_date: Minimum date for the time series
    :param max_date: Maximum date for the time series
    :param date_format: date format for faster conversion to datetime (optional)
    :return: a time-series DataFrame
    """
    if "date" not in raw:
        raise ValueError("File must contain a date column")

    raw["date"] = pd.to_datetime(raw["date"], format=date_format)
    if min_date is not None:
        raw = raw[raw["date"] >= min_date]
    if max_date is not None:
        raw = raw[raw["date"] <= max_date]

    raw.set_index(pd.DatetimeIndex(raw["date"]), inplace=True, drop=True)
    if "date" in raw:
        raw.drop(columns="date", inplace=True)
    raw.loc[:, "timestamp"] = raw.index.astype(np.int64) // 10 ** 9
    return raw


def ts_to_range(ts, time_range="1H", pad_with_zeros=True):
    """
    Creates a new data frame with counts per time range
    :param ts: The original time series, with a column named 'date' and possibly an additional 'category' column
    :param time_range: The time range requested
    :param pad_with_zeros: Whether to add a value of 0 for missing dates (see Pandas resample).
    Note that this will split the time series into different categories (if categories exist), and pad each category
    independently. Then all time series will be appended together.
    :return: a pd.DataFrame with a MultiIndex containing a date and category. Contains an additional column with
    counts in the interval
    """
    if not isinstance(ts.index, pd.core.indexes.datetimes.DatetimeIndex):
        print("Wrong index type. Expecting pd.core.indexes.datetimes.DateTimeIndex")
        return

    if "category" in ts:
        range_grp = (
            ts.groupby([pd.Grouper(freq=time_range), "category"])
            .size()
            .to_frame("value")
        )
        range_grp = range_grp.reset_index(level="category")
        if pad_with_zeros:
            categories = range_grp["category"].unique()
            new_range_grp = pd.DataFrame()
            for cat in categories:
                this_cat = range_grp[range_grp["category"] == cat]
                this_cat = (
                    this_cat.resample(time_range, convention="start").asfreq().fillna(0)
                )
                this_cat["category"] = cat
                new_range_grp = new_range_grp.append(this_cat, ignore_index=False)

            range_grp = new_range_grp.set_index(["category"], append=True).sort_index(0)

    else:
        range_grp = ts.groupby(pd.Grouper(freq=time_range)).size().to_frame("value")
        if pad_with_zeros:
            range_grp = (
                range_grp.resample(time_range, convention="start").asfreq().fillna(0)
            )

    return range_grp
