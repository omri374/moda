"""Test evaluation functionality."""
import math
import os

import pandas as pd
import pytest

from moda.models.data_reader import read_data
from moda.evaluators.eval import eval_models, f_beta, get_metrics_with_shift, eval_models_CV
from moda.models.ma_seasonal import MovingAverageSeasonalTrendinessDetector
from moda.test.mock_model import MockModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_read_data():
	path = os.path.join(THIS_DIR, os.pardir, 'test/dummy.txt')
	df = read_data(path)
	index = df.index
	# assert index is pd.MultiIndex
	assert len(df) == 16
	assert isinstance(df.index, pd.MultiIndex)


def test_eval_models_all_true():
	path = os.path.join(THIS_DIR, os.pardir, 'test/dummy2.txt')
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
	path = os.path.join(THIS_DIR, os.pardir, 'test/dummy.txt')
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
	path = os.path.join(THIS_DIR, os.pardir, 'test/dummy2.txt')
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
	path = os.path.join(THIS_DIR, os.pardir, 'test/dummy.txt')
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
	path = os.path.join(THIS_DIR, os.pardir, 'test/dummy.txt')
	df = read_data(path)
	X = df[['value']]
	y = None

	model = MockModel()
	models = [model]
	with pytest.raises(TypeError):
		eval_models(X, y, models, label_col_name='is_anomaly')


def test_n_splits_big():
	path = os.path.join(THIS_DIR, os.pardir, 'test/dummy2.txt')
	df = read_data(path)

	model = MockModel()
	models = [model]
	X = df[['value']]
	y = df[['is_anomaly']]
	res = eval_models_CV(X, y, models, n_splits=40000, verbose=True, label_col_name='is_anomaly')
	assert res['MockModel']['f1'] == 1.0
	assert res['MockModel']['precision'] == 1.0
	assert res['MockModel']['recall'] == 1.0


def test_f_beta1():
	precision = 0.6
	recall = 1.0
	beta = 1
	f = f_beta(precision, recall, beta)
	assert (f > 0.74) and (f < 0.76)


def test_f_beta3():
	precision = 0.6
	recall = 1.0
	beta = 3
	f = f_beta(precision, recall, beta)
	assert (f > 0.937) and (f < 0.938)


def test_get_metrics_with_shift_all_zero():
	actual = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	predicted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=1)
	assert metrics['cat']['TP'] == 0
	assert metrics['cat']['FP'] == 0
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 0


def test_get_metrics_with_shift_actual_zero():
	actual = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	predicted = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=1)
	assert metrics['cat']['TP'] == 0
	assert metrics['cat']['FP'] == 2
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 0

	predicted = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=1)
	assert metrics['cat']['TP'] == 0
	assert metrics['cat']['FP'] == 4
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 0


def test_get_metrics_with_shift_predicted_zero():
	actual = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
	predicted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=1)
	assert metrics['cat']['TP'] == 0
	assert metrics['cat']['FP'] == 0
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 2

	actual = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=1)
	assert metrics['cat']['TP'] == 0
	assert metrics['cat']['FP'] == 0
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 4


def test_get_metrics_with_shift_perfect():
	actual = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
	predicted = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=1)
	assert metrics['cat']['TP'] == 2
	assert metrics['cat']['FP'] == 0
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 0

	actual = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
	predicted = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=1)
	assert metrics['cat']['TP'] == 4
	assert metrics['cat']['FP'] == 0
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 0


def test_get_metrics_with_shift_mixed():
	actual = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
	predicted = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=1)
	assert metrics['cat']['TP'] == 1
	assert metrics['cat']['FP'] == 1
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 1


def test_get_metrics_with_shift_in_window():
	actual = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
	predicted = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=1)
	assert metrics['cat']['TP'] == 1
	assert metrics['cat']['FP'] == 0
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 0


def test_get_metrics_with_shift_in_large_window():
	actual = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
	predicted = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	metrics = get_metrics_with_shift(predicted, actual, 'cat', window_size=4)
	assert metrics['cat']['TP'] == 1
	assert metrics['cat']['FP'] == 0
	assert metrics['cat']['TN'] == 0
	assert metrics['cat']['FN'] == 0


if __name__ == '__main__':
	test_read_data()
	test_eval_models_all_true()
	test_eval_models_all_false()
	test_eval_models_half_false()
	test_f_beta1()
	test_f_beta3()
	test_X_None()
	test_y_None()
	test_n_splits_big()
	test_get_metrics_with_shift_all_zero()
	test_get_metrics_with_shift_actual_zero()
	test_get_metrics_with_shift_predicted_zero()
	test_get_metrics_with_shift_perfect()
	test_get_metrics_with_shift_mixed()
	test_get_metrics_with_shift_in_window()
	test_get_metrics_with_shift_in_large_window()
	test_real_model()
