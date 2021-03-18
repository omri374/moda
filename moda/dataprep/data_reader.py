import pandas as pd


def read_data(path: str, min_date=None, max_date=None) -> pd.DataFrame:
    """Reads data to be fed into the model.

    :param path: Path to the CSV fie containing the data to be modeled.
        The file should contain a date/time column,
        a category column and the value for this category and this date/time.

    :param min_date: Minimum date to return data for. Example: "01-01-2018"

    :param max_date: Maximum date to return data for. Example: "01-01-2019"

    :returns A pandas DataFrame with a two-leveled multi-index, the first
           indexing time and the second indexing class/topic frequency
           per-window, and a single column of a numeric dtype, giving said
           frequency.
    """

    import os.path

    if not os.path.exists(path):
        raise FileNotFoundError

    df = pd.read_csv(path, index_col=None)

    df["date"] = pd.to_datetime(df["date"])
    if min_date is not None:
        df = df[df["date"] >= min_date]
    if max_date is not None:
        df = df[df["date"] <= max_date]

    df = df.set_index([pd.DatetimeIndex(df["date"]), "category"]).drop(columns="date")
    print("Read {0} rows".format(len(df)))
    return df
