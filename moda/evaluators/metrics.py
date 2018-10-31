import numpy as np
import pandas as pd


def _initialize_metrics_dict():
    """
    Initialize the metrics dictionary for one category
    :return: An empty metrics dictionary
    """
    metrics = {'TP': 0, 'FP': 0, 'FN': 0, 'num_samples': 0, 'num_values': 0}
    return metrics


def _initialize_metrics(categories):
    """
    Initialize the metrics dictionary for all categories
    :param categories: A list of categories
    :return: An empty metrics dictionary
    """
    metrics = {}
    for category in categories:
        metrics[category] = _initialize_metrics_dict()

    return metrics


def calculate_metrics_with_shift(predicted, actual, window_size=3):
    """
    Calculates TP, FP, and FN while allowing shifts. For example,
    predicted = [0,1,0,0], actual [1,0,0,0] and a window_size of 1 would return a TP,
    whereas predicted [0,1,0,0] and actual [1,0,0,0] with window size 0 would return both a FP and FN
    :param predicted: an array of 0 and 1 values of predictions
    :param actual:  an array of 0 and 1 values of actual values
    :param window_size: The allowed shift to the left or the right.
    a window_size of 2 means that the corresponding value will be looked for
    in 2 cells to the left and two cells to the right
    :return: metrics, a dictionary holding the TP, FP and FN values per category.
    If a metrics value is provided as input, values will be aggregated on top of the existing metrics dictionary
    """
    metrics = _initialize_metrics_dict()
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
                    metrics['TP'] += 1
                else:
                    metrics['FN'] += 1
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
                    metrics['FP'] += 1
            prev = pred
    return metrics


def _get_metrics_for_one_category(dataset, label_col_name, prediction_col_name, value_col_name,
                                  window_size_for_metrics, category="", prev_metrics=None):
    """
    Returns metrics for a specific category in the data.
    :param dataset: A dataset holding the prediction, actual and value columns.
    :param label_col_name: The name of the actual values (labeled) Series
    :param prediction_col_name: The name of the predicted values Series
    :param value_col_name: Name of the value column in the dataset
    :param window_size_for_metrics:  The allowed shift to the left or the right.
    a window_size of 2 means that the corresponding value will be looked for
    in 2 cells to the left and two cells to the right
    :param category: (Optional) The name of the category if the dataset holds multiple categories
    :param prev_metrics: A dictionary of TP, FP and FN, if exists from a previous iteration

    :return: a dictionary with the precision, recall, f1, f0.5 metrics,
    as well as the number of samples per category and the sum of values.
    """
    category_results = dataset.loc[pd.IndexSlice[:, category], :]
    category_results.index = category_results.index.remove_unused_levels()
    # Calculate TP, FP and FN
    new_metrics = calculate_metrics_with_shift(predicted=category_results[prediction_col_name].values,
                                               actual=category_results[label_col_name].values,
                                               window_size=window_size_for_metrics)
    # Calculate additional accumulators
    new_metrics['num_samples'] += len(category_results)
    new_metrics['num_values'] += np.sum(category_results[value_col_name])
    if 'TP' not in new_metrics:
        new_metrics['TP']=np.NaN
    if 'FP' not in new_metrics:
        new_metrics['FP'] = np.NaN
    if 'FN' not in new_metrics:
        new_metrics['FN'] = np.NaN
    new_metrics = _join_metrics(new_metrics, prev_metrics)

    return new_metrics


def _join_pred_to_dataset(original_df, prediction_df, test_values_df, label_col_name):
    results = pd.merge(prediction_df, original_df, how='left', on=['date', 'category'])[['prediction', label_col_name]]
    results = pd.merge(results, test_values_df, how='left', on=['date', 'category'])
    results[label_col_name] = results[label_col_name].fillna(0)
    results.sort_index(level=['date', 'category'], ascending=True, inplace=True)
    return results


def get_metrics_for_all_categories(test_values_df, prediction_df, labels_df, value_col_name='value',
                                   label_col_name='label', prediction_col_name='prediction',
                                   window_size_for_metrics=5):
    """
    Evalutes a model with a specific set of raw_metrics,
    :param label_col_name: The name of the label column in the testing data
    :param raw_metrics: A dictionary of metric names and current values (i.e. {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'num_samples': 0, 'num_values': 0})
    :param model: A model to be run, which has a fit and predict methods.
    :param prediction_col_name:
    :param test_labels: The test set's labels Series
    :param test_samples: The test set
    :param train_labels: The training set's labels Series
    :param train_samples: The training set
    :param value_col_name: The name of the Series with the actual time series values
    :param verbose:
    """

    categories = prediction_df.index.levels[1]
    all_metrics = {}

    # join the model results and test set, assuming that some dates or times are missing
    dataset = _join_pred_to_dataset(labels_df, prediction_df, test_values_df, label_col_name)
    # print(results)
    for category in dataset.index.levels[1]:
        cat_metrics = _get_metrics_for_one_category(dataset, label_col_name, prediction_col_name,
                                                    value_col_name, window_size_for_metrics, category)
        all_metrics[category] = cat_metrics

    return all_metrics


def _join_metrics(metrics1, metrics2):
    if metrics1 is None:
        return metrics2
    if metrics2 is None:
        return metrics1

    for category in metrics1:
        if category in metrics2:
            metrics1[category] = _join_raw_metrics(metrics1[category], metrics2[category])

    for category in metrics2:
        if category not in metrics1:
            metrics1[category] = {}
            for metric in metrics2[category]:
                metrics1[category][metric] = metrics2[category][metric]

    return metrics1


def _join_raw_metrics(raw1, raw2):
    raw = {}
    raw['TP'] = raw1['TP'] + raw2['TP']
    raw['FP'] = raw1['FP'] + raw2['FP']
    raw['FN'] = raw1['FN'] + raw2['FN']
    raw['num_samples'] = raw1['num_samples'] + raw2['num_samples']
    raw['num_values'] = raw1['num_values'] + raw2['num_values']

    return raw


def get_final_metrics(raw_metrics, summarized=False):
    """
    Calculates final metrics from all categories.
    :param summarized: True if the result should contain only final metrics (precision recall, f1 and f0.5)
    False if the result should contain all the per category metrics too.
    :param raw_metrics: A dictionary of tp, fp and fn values for each category
    :return: a dictionary with the precision, recall, f1 and f0.5 metrics, as well as the input metrics data.
    """

    tp = 0
    fp = 0
    fn = 0
    num_values = 0
    num_samples = 0
    final_metrics = dict()


    for category in raw_metrics:
        category_tp = raw_metrics[category]['TP']
        category_fp = raw_metrics[category]['FP']
        category_fn = raw_metrics[category]['FN']

        final_metrics[category] = {}

        if category_tp > 0:
            final_metrics[category]['precision'] = category_tp / (category_tp + category_fp)
            final_metrics[category]['recall'] = category_tp / (category_tp + category_fn)
            final_metrics[category]['f1'] = f_beta(final_metrics[category]['precision'],
                                                     final_metrics[category]['recall'], 1


                                                     )
        if 'num_values' in raw_metrics[category]:
            final_metrics[category]['num_values'] = raw_metrics[category]['num_values']
        if 'num_samples' in raw_metrics[category]:
            final_metrics[category]['num_samples'] = raw_metrics[category]['num_samples']

        tp += category_tp
        fp += category_fp
        fn += category_fn
        num_values += final_metrics[category]['num_values']
        num_samples += final_metrics[category]['num_samples']

    if (tp + fp) > 0:
        final_metrics['precision'] = tp / (tp + fp)
    else:
        final_metrics['precision'] = np.nan
    if (tp + fn) > 0:
        final_metrics['recall'] = tp / (tp + fn)
    else:
        final_metrics['recall'] = np.nan
    final_metrics['f1'] = f_beta(final_metrics['precision'], final_metrics['recall'], 1)
    final_metrics['f0.5'] = f_beta(final_metrics['precision'], final_metrics['recall'], 0.5)
    final_metrics['num_values'] = num_values
    final_metrics['num_samples'] = num_samples

    if summarized:
        return summarize_metrics(final_metrics)
    else:
        return final_metrics


def summarize_metrics(all_metrics):
    """
    Returns a subset of the metrics dictionary with only f1,f0.5,recall and precision values
    :param all_metrics:
    :return:
    """
    metric_names = ['f1', 'recall', 'precision', 'f0.5', 'num_samples','num_metrics']  # The keys to keep
    return dict((k, all_metrics[k]) for k in metric_names if k in all_metrics)


def f_beta(precision, recall, beta):
    """
    Returns the F score for precision, recall and a beta parameter
    :param precision: a double with the precision value
    :param recall: a double with the recall value
    :param beta: a double with the beta parameter of the F measure, which gives more or less weight to precision vs. recall
    :return: a double value of the f(beta) measure.
    """
    if np.isnan(precision) or np.isnan(recall) or (precision == 0 and recall == 0):
        return np.nan
    return ((1 + beta ** 2) * precision * recall) / (((beta ** 2) * precision) + recall)
