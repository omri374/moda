"""Test stl trend detector functionality."""
import os

import pandas as pd

from moda.models import read_data
from moda.models.ma_seasonal import MovingAverageSeasonalTrendinessDetector

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_category():
    path = os.path.join(THIS_DIR, os.pardir, 'test/dummy3.txt')
    df = read_data(path)
    one_category = df.loc[pd.IndexSlice[:, 'housing'], :].reset_index(level='category',drop=True)
    stl = MovingAverageSeasonalTrendinessDetector(freq='12H', is_multicategory=False)
    prediction = stl.predict(one_category)
    print(prediction)


if __name__ == '__main__':
    test_single_category()
