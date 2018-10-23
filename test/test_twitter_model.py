"""Test stl trend detector functionality."""
import os

import pandas as pd
from moda.models.twitter import TwitterAnomalyTrendinessDetector

from moda.dataprep.data_reader import read_data

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_category():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy4.txt')
    df = read_data(path)
    one_category = df.loc[pd.IndexSlice[:, 'housing'], :].reset_index(level='category', drop=True)
    twitter = TwitterAnomalyTrendinessDetector(freq='12H', is_multicategory=False)
    prediction = twitter.predict(one_category)
    assert len(prediction[prediction['prediction'] == 1]) == 1


def test_two_categories_high_std():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy4.txt')
    df = read_data(path)
    twitter = TwitterAnomalyTrendinessDetector(freq='12H', is_multicategory=False)
    prediction = twitter.predict(df)
    assert len(prediction[prediction['prediction'] == 1]) == 0

def test_two_categories_high_min_value():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy4.txt')
    df = read_data(path)
    twitter = TwitterAnomalyTrendinessDetector(freq='12H', is_multicategory=False)
    prediction = twitter.predict(df)
    assert len(prediction[prediction['prediction'] == 1]) == 0


if __name__ == '__main__':
    test_single_category()
    test_two_categories_high_std()
    test_two_categories_high_min_value()
