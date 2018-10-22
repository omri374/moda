"""Test evaluation functionality."""
import math
import os

import pandas as pd
import pytest

from moda.evaluators.eval import eval_models, eval_models_CV
from moda.dataprep.data_reader import read_data
from moda.models.ma_seasonal import MovingAverageSeasonalTrendinessDetector
from tests.mock_model import MockModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_read_data():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy.txt')
    df = read_data(path)
    index = df.index
    # assert index is pd.MultiIndex
    assert len(df) == 16
    assert isinstance(df.index, pd.MultiIndex)


def test_eval_models_all_true():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy2.txt')
    df = read_data(path)

    model = MockModel()
    models = [model]
    X = df[['value']]
    y = df[['is_anomaly']]
    res = eval_models(X, y, models, label_col_name='is_anomaly')
    print(res)
    assert res['MockModel']['f1'] == 1.0
    assert res['MockModel']['precision'] == 1.0
    assert res['MockModel']['recall'] == 1.0


def test_eval_models_all_false():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy.txt')
    df = read_data(path)
    df['is_anomaly'] = 0

    model = MockModel()
    models = [model]
    X = df[['value']]
    y = df[['is_anomaly']]
    res = eval_models(X, y, models, label_col_name='is_anomaly')
    assert math.isnan(res['MockModel']['f1'])
    assert res['MockModel']['precision'] == 0
    assert math.isnan(res['MockModel']['recall'])


def test_eval_models_half_false():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy2.txt')
    df = read_data(path)
    df['is_anomaly'] = 0
    df.iloc[-1]['is_anomaly'] = 1
    df.iloc[-2]['is_anomaly'] = 1

    model = MockModel()
    models = [model]
    X = df[['value']]
    y = df[['is_anomaly']]
    res = eval_models(X, y, models, label_col_name='is_anomaly', window_size_for_metrics=0)
    assert res['MockModel']['precision'] == 0.5
    assert res['MockModel']['recall'] == 1.0


def test_real_model():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy.txt')
    df = read_data(path)
    model = MovingAverageSeasonalTrendinessDetector(is_multicategory=True, freq='12H')
    models = [model]
    X = df[['value']]
    y = df[['is_anomaly']]

    res = eval_models(X, y, models, label_col_name='is_anomaly')
    print(res)


def test_X_None():
    X = None
    y = None
    model = MockModel()
    models = [model]
    try:
        res = eval_models(X, y, models, label_col_name='is_anomaly')
    except TypeError:
        assert True
        return
    assert False


def test_y_None():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy.txt')
    df = read_data(path)
    X = df[['value']]
    y = None

    model = MockModel()
    models = [model]
    with pytest.raises(TypeError):
        eval_models(X, y, models, label_col_name='is_anomaly')


def test_n_splits_big():
    path = os.path.join(THIS_DIR, os.pardir, 'test/data/dummy2.txt')
    df = read_data(path)

    model = MockModel()
    models = [model]
    X = df[['value']]
    y = df[['is_anomaly']]
    res = eval_models_CV(X, y, models, n_splits=40000, verbose=True, label_col_name='is_anomaly')
    assert res['MockModel']['f1'] == 1.0
    assert res['MockModel']['precision'] == 1.0
    assert res['MockModel']['recall'] == 1.0


if __name__ == '__main__':
    test_read_data()
    test_eval_models_all_true()
    test_eval_models_all_false()
    test_eval_models_half_false()
    test_X_None()
    test_y_None()
    test_n_splits_big()
    test_real_model()
