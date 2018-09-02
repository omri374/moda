import os

import pandas as pd

from moda.dataprep.raw_to_ts import raw_to_ts
from moda.dataprep.ts_to_range import ts_to_range

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_ts_to_range():
    path = os.path.join(THIS_DIR, os.pardir, 'test/sample-raw.csv')
    raw = pd.read_csv(path)
    ts = raw_to_ts(raw)
    range = ts_to_range(ts, time_range='24H')
    assert 'category' in range.index.names
    assert 'date' in range.index.names


def test_ts_to_range_no_categories():
    path = os.path.join(THIS_DIR, os.pardir, 'test/sample-raw.csv')
    raw = pd.read_csv(path)
    ts = raw_to_ts(raw)
    ts2 = ts[ts['category'] == 'A']
    ts2 = ts2.drop(columns='category')
    range = ts_to_range(ts2, time_range='24H')
    assert 'category' not in range.index.names
    assert 'date' in range.index.names


if __name__ == '__main__':
    test_ts_to_range()
    test_ts_to_range_no_categories()
