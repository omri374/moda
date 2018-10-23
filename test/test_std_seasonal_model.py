"""Test stl trend detector functionality."""
import os

import pandas as pd
from moda.dataprep.data_reader import read_data
from moda.models.stl.stl_model import STLTrendinessDetector

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_category():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy4.txt')
    df = read_data(path)
    one_category = df.loc[pd.IndexSlice[:, 'housing'], :].reset_index(level='category', drop=True)
    ma = STLTrendinessDetector(freq='12H', is_multicategory=False, anomaly_type='or')
    prediction = ma.predict(one_category)
    assert len(prediction[prediction['prediction'] == 1]) == 1


def test_two_categories():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy4.txt')
    df = read_data(path)
    ma = STLTrendinessDetector(freq='12H', anomaly_type='or', lookback='5D')
    prediction = ma.predict(df)
    assert len(prediction[prediction['prediction'] == 1]) == 2


def test_two_categories_high_std():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy4.txt')
    df = read_data(path)
    ma = STLTrendinessDetector(freq='12H', anomaly_type='or', lookback='5D',num_of_std=50)
    prediction = ma.predict(df)
    assert len(prediction[prediction['prediction'] == 1]) == 0

def test_two_categories_high_min_value():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy4.txt')
    df = read_data(path)
    ma = STLTrendinessDetector(freq='12H', anomaly_type='or', lookback='5D',min_value=2000)
    prediction = ma.predict(df)
    assert len(prediction[prediction['prediction'] == 1]) == 0


if __name__ == '__main__':
    test_single_category()
    test_two_categories()
    test_two_categories_high_std()
    test_two_categories_high_min_value()
