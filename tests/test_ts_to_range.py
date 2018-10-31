import os

import pandas as pd

from moda.dataprep import raw_to_ts
from moda.dataprep.ts_to_range import ts_to_range

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_ts_to_range():
    path = os.path.join(THIS_DIR, os.pardir, 'tests/data/sample-raw.csv')
    raw = pd.read_csv(path).sort_values('date')
    ts = raw_to_ts(raw)
    range = ts_to_range(ts, time_range='24H')
    assert 'category' in range.index.names
    assert 'date' in range.index.names
    assert range.iloc[0,0] > 1


def test_ts_to_range_no_categories():
    path = os.path.join(THIS_DIR, os.pardir, 'tests/data/sample-raw.csv')
    raw = pd.read_csv(path)
    ts = raw_to_ts(raw)
    ts2 = ts[ts['category'] == 'A']
    ts2 = ts2.drop(columns='category')
    range = ts_to_range(ts2, time_range='24H')
    assert 'category' not in range.index.names
    assert 'date' in range.index.names
    assert range.iloc[0,0] > 1


if __name__ == '__main__':
    test_ts_to_range()
    test_ts_to_range_no_categories()
