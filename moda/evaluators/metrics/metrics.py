import numpy as np


def get_metrics_with_shift(predicted, actual, category="", metrics=None, window_size=5):
    """
    Calculates TP, FP, and FN while allowing shifts. For example, predicted = [0,1,0], actual [1,0,0] and a window_size of 1 would return a TP.
    :param predicted: an array of 0 and 1 values of predictions
    :param actual:  an array of 0 and 1 values of actual values
    :param category: The current category evaluated, in case of multiple categories
    :param metrics: A dictionary of TP, FP and FN per category
    :param window_size: The allowed shift to the left or the right.
    a window_size of 2 means that the corresponding value will be looked for
    in 2 cells to the left and two cells to the right
    :return: metrics, a dictionary holding the TP, FP and FN values per category.
    If a metrics value is provided as input, values will be aggregated on top of the existing metrics dictionary
    """
    if metrics is None:
        metrics = {}
    if metrics.get(category) is None:
        metrics[category] = _initialize_metrics_one_category()
    n = len(predicted)

    # Iterate over all labels, and look for corresponding predictions in the window.
    prev = -1

    if ~np.all(actual == 0):
        for idx, act in enumerate(actual):
            found = False
            if actual[idx] == 1 and actual[idx] != prev:
                for size in range(-window_size, window_size + 1):
                    if n > (idx + size) >= 0:
                        if predicted[idx + size] == 1:
                            found = True

                if found:
                    metrics[category]['TP'] += 1
                else:
                    metrics[category]['FN'] += 1
            prev = act

    # Iterate over all positive predictions, and look for corresponding labels in the window
    prev = -1
    if ~np.all(predicted == 0):
        for idx, pred in enumerate(predicted):
            found = False
            if pred == 1 and pred != prev:
                for size in range(-window_size, window_size + 1):
                    if n > (idx + size) >= 0:
                        if actual[idx + size] == 1:
                            found = True

                if not found:
                    metrics[category]['FP'] += 1
            prev = pred
    return metrics


def _initialize_metrics_one_category():
    """
    Initialize the metrics dictionary for one category
    :return: An empty metrics dictionary
    """
    metrics = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'num_samples': 0, 'num_values': 0}
    return metrics


def _initialize_metrics(categories):
    """
    Initialize the metrics dictionary for one category
    :param categories: A list of categories
    :return: An empty metrics dictionary
    """
    metrics = {}
    for category in categories:
        metrics[category] = _initialize_metrics_one_category()
    return metrics


def get_all_metrics(metrics):
    """
    Aggregates metrics across all categories.
    :param metrics: A dictionary of TP,FP,TN and FN values for each category
    :return: a dictionary with the precision, recall, f1 and f0.5 metrics, as well as the input metrics data.
    """
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for category in metrics:
        TP += np.sum(metrics[category]['TP'])
        FP += np.sum(metrics[category]['FP'])
        TN += np.sum(metrics[category]['TN'])
        FN += np.sum(metrics[category]['FN'])

    final_metrics = dict()
    if (TP + FP) > 0:
        final_metrics['precision'] = TP / (TP + FP)
    else:
        final_metrics['precision'] = np.nan
    if (TP + FN) > 0:
        final_metrics['recall'] = TP / (TP + FN)
    else:
        final_metrics['recall'] = np.nan
    final_metrics['f1'] = f_beta(final_metrics['precision'], final_metrics['recall'], 1)
    final_metrics['f0.5'] = f_beta(final_metrics['precision'], final_metrics['recall'], 0.5)
    final_metrics['raw'] = metrics
    return final_metrics


def get_final_metrics(metrics):
    if metrics is None:
        return None

    final_metrics = ['f1', 'recall', 'precision', 'f0.5']  # The keys you want
    return dict((k, metrics[k]) for k in final_metrics if k in metrics)


def f_beta(precision, recall, beta):
    """
    Returns the F score for precision, recall and a beta parameter
    :param precision: a double with the precision value
    :param recall: a double with the recall value
    :param beta: a double with the beta parameter of the F measure, which gives more or less weight to precision vs. recall
    :return: a double value of the f(beta) measure.
    """
    if np.isnan(precision) or np.isnan(recall):
        return np.nan
    return ((1 + beta ** 2) * precision * recall) / (((beta ** 2) * precision) + recall)
