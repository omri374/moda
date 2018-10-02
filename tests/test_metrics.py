"""Test evaluation functionality."""
import os

from moda.evaluators.eval import get_metrics_with_shift
from moda.evaluators.metrics.metrics import f_beta

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
    test_f_beta1()
    test_f_beta3()
    test_get_metrics_with_shift_all_zero()
    test_get_metrics_with_shift_actual_zero()
    test_get_metrics_with_shift_predicted_zero()
    test_get_metrics_with_shift_perfect()
    test_get_metrics_with_shift_mixed()
    test_get_metrics_with_shift_in_window()
    test_get_metrics_with_shift_in_large_window()
