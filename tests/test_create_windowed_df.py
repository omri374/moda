import pandas as pd

from moda.dataprep.create_dataset import get_windowed_ts


def test_create_windowed_df():
    ranged_ts = pd.DataFrame({'date': range(6), 'value': range(6)})
    ranged_ts['date'] = pd.to_datetime(ranged_ts['date'])
    ranged_ts = ranged_ts.set_index(pd.DatetimeIndex(ranged_ts['date']))
    ranged_ts = ranged_ts.drop(columns='date')
    windowed_df = get_windowed_ts(ranged_ts, window_size=3, with_actual=False)

    assert len(windowed_df.columns) == 3

    assert windowed_df.iloc[0, 0] == 0
    assert windowed_df.iloc[0, 1] == 1
    assert windowed_df.iloc[0, 2] == 2

    assert windowed_df.iloc[1, 0] == 1
    assert windowed_df.iloc[1, 1] == 2
    assert windowed_df.iloc[1, 2] == 3
