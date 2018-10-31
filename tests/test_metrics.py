"""Test evaluation functionality."""
from moda.evaluators import f_beta
from moda.evaluators.metrics import calculate_metrics_with_shift, _join_metrics


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


def test_calculate_metrics_with_shift_all_zero():
    actual = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    predicted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=1)
    assert metrics['TP'] == 0
    assert metrics['FP'] == 0
    assert metrics['FN'] == 0


def test_calculate_metrics_with_shift_actual_zero():
    actual = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    predicted = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=1)
    assert metrics['TP'] == 0
    assert metrics['FP'] == 2
    assert metrics['FN'] == 0

    predicted = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=1)
    assert metrics['TP'] == 0
    assert metrics['FP'] == 4
    assert metrics['FN'] == 0


def test_calculate_metrics_with_shift_predicted_zero():
    actual = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    predicted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=1)
    assert metrics['TP'] == 0
    assert metrics['FP'] == 0
    assert metrics['FN'] == 2

    actual = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=1)
    assert metrics['TP'] == 0
    assert metrics['FP'] == 0
    assert metrics['FN'] == 4


def test_calculate_metrics_with_shift_perfect():
    actual = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    predicted = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=1)
    assert metrics['TP'] == 2
    assert metrics['FP'] == 0
    assert metrics['FN'] == 0

    actual = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    predicted = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=1)
    assert metrics['TP'] == 4
    assert metrics['FP'] == 0
    assert metrics['FN'] == 0


def test_calculate_metrics_with_shift_mixed():
    actual = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    predicted = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=1)
    assert metrics['TP'] == 1
    assert metrics['FP'] == 1
    assert metrics['FN'] == 1


def test_calculate_metrics_with_shift_in_window():
    actual = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    predicted = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=1)
    assert metrics['TP'] == 1
    assert metrics['FP'] == 0
    assert metrics['FN'] == 0


def test_calculate_metrics_with_shift_in_large_window():
    actual = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    predicted = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    metrics = calculate_metrics_with_shift(predicted, actual, window_size=4)
    assert metrics['TP'] == 1
    assert metrics['FP'] == 0
    assert metrics['FN'] == 0


def test_join_metrics():
    metrics1 = {'cat1': {'TP': 12, 'FP': 7, 'FN': 35, 'num_samples': 1, 'num_values': 2},
                'cat2': {'TP': 0, 'FP': 0, 'FN': 0, 'num_samples': 1, 'num_values': 2},
                'cat4': {'TP': 9, 'FP': 9, 'FN': 9, 'num_samples': 1, 'num_values': 2}}

    metrics2 = {'cat1': {'TP': 10, 'FP': 10, 'FN': 10, 'num_samples': 1, 'num_values': 2},
                'cat2': {'TP': 2, 'FP': 2, 'FN': 2, 'num_samples': 1, 'num_values': 2},
                'cat3': {'TP': 1, 'FP': 1, 'FN': 1, 'num_samples': 1, 'num_values': 2}}

    metrics = _join_metrics(metrics1, metrics2)

    assert metrics['cat1']['TP'] == 22
    assert metrics['cat1']['FP'] == 17
    assert metrics['cat1']['FN'] == 45
    assert metrics['cat1']['num_samples'] == 2
    assert metrics['cat1']['num_values'] == 4

    assert metrics['cat2']['TP'] == 2
    assert metrics['cat2']['FP'] == 2
    assert metrics['cat2']['FN'] == 2

    assert metrics['cat3']['TP'] == 1
    assert metrics['cat3']['FP'] == 1
    assert metrics['cat3']['FN'] == 1

    assert metrics['cat4']['TP'] == 9
    assert metrics['cat4']['FP'] == 9
    assert metrics['cat4']['FN'] == 9


if __name__ == '__main__':
    test_f_beta1()
    test_f_beta3()
    test_calculate_metrics_with_shift_all_zero()
    test_calculate_metrics_with_shift_actual_zero()
    test_calculate_metrics_with_shift_predicted_zero()
    test_calculate_metrics_with_shift_perfect()
    test_calculate_metrics_with_shift_mixed()
    test_calculate_metrics_with_shift_in_window()
    test_calculate_metrics_with_shift_in_large_window()
    test_join_metrics()
