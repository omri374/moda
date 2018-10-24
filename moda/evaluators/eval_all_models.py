import json
import os

import numpy as np
from comet_ml import Experiment

from moda.dataprep import read_data
from moda.evaluators import eval_models, get_metrics_for_all_categories, summarize_metrics, get_final_metrics, \
    eval_models_CV
from moda.models import AzureAnomalyTrendinessDetector, LSTMTrendinessDetector
from moda.models import MovingAverageSeasonalTrendinessDetector
from moda.models import STLTrendinessDetector
from moda.models import TwitterAnomalyTrendinessDetector


def evaluate_all_models(datapath="SF3H_labeled.csv", min_date='01-01-2018', freq='3H', use_comet=True, models_to_run=[],
                        window_size_for_metrics=5):
    try:
        dataset = read_data(datapath, min_date=min_date)
    except:
        print("File not found or failed to read")
        return
    dataset = dataset[~dataset.index.duplicated(keep='first')]
    dataset = dataset.rename(columns={'is_anomaly': 'label'})

    X = dataset[['value']]
    y = dataset[['label']]

    if 'H' in freq:
        min_value = 10
    else:
        min_value = 8

    print("min value for prediction = " + str(min_value))

    # LSTM model
    if 'lstm' in models_to_run:
        print("Evaluating LSTM model")
        num_std = 3
        model = LSTMTrendinessDetector(is_multicategory=True, num_of_std=num_std, freq=freq, min_value=min_value)
        result = eval_models(X, y, [model], label_col_name='label', train_percent=20,
                             window_size_for_metrics=window_size_for_metrics)

        print_lstm_model(datapath, min_value, model, result)

        if use_comet:
            params = {'num_std': num_std,
                      'window_size_for_metrics': window_size_for_metrics
                      }
            metrics = result[model.__name__]
            metrics = summarize_metrics(metrics)
            log_experiment(datapath, dataset, model, parameters=params, metrics=metrics)

    # MA model
    if 'ma_seasonal' in models_to_run or len(models_to_run) == 0:
        print("Evaluating MA model")
        anomaly_types = ['residual', 'trend', 'and', 'or']
        for num_std in [2.5, 3, 3.5, 4]:
            for anomaly_type in anomaly_types:
                model = MovingAverageSeasonalTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                                                anomaly_type=anomaly_type, num_of_std=num_std)
                result = eval_models_CV(X, y, [model], label_col_name='label', n_splits=5,
                                        window_size_for_metrics=window_size_for_metrics)
                print_ma_result(anomaly_type, datapath, min_value, model, num_std, result)

                if use_comet:
                    params = {'anomaly_type': anomaly_type,
                              'num_std': num_std,
                              'window_size_for_metrics': window_size_for_metrics,
                              'min_value': min_value
                              }
                    metrics = result[model.__name__]
                    metrics = summarize_metrics(metrics)

                log_experiment(datapath, dataset, model, parameters=params, metrics=metrics)

    # STL model
    if 'stl' in models_to_run or len(models_to_run) == 0:
        print("Evaluating STL model")
        for num_std in [2.5, 3, 3.5, 4]:
            for anomaly_type in anomaly_types:
                for lo_frac in [0.1, 0.5, 1, 1.5]:
                    model = STLTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                                  anomaly_type=anomaly_type, num_of_std=num_std, lo_frac=lo_frac)
                    result = eval_models(X, y, [model], label_col_name='label', train_percent=20,
                                         window_size_for_metrics=window_size_for_metrics)
                    print_stl_result(anomaly_type, datapath, min_value, model, num_std, result)

                    if use_comet:
                        params = {'anomaly_type': anomaly_type,
                                  'num_std': num_std,
                                  'window_size_for_metrics': window_size_for_metrics,
                                  'min_value': min_value,
                                  'lo_frac': lo_frac
                                  }
                        metrics = result[model.__name__]
                        metrics = summarize_metrics(metrics)
                        log_experiment(datapath, dataset, model, parameters=params, metrics=metrics)
    # Twitter
    if 'twitter' in models_to_run or len(models_to_run) == 0:
        print("Evaluating Twitter model")
        max_anoms_list = [0.05, 0.1]
        for max_anoms in max_anoms_list:
            for threshold in [None, 'med_max', 'p95', 'p99']:
                for alpha in [0.05, 0.1, 0.15]:
                    model = TwitterAnomalyTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                                             threshold=threshold, max_anoms=max_anoms,
                                                             longterm=False, alpha=alpha, seasonality_freq=7)

                    result = eval_models(X, y, [model], label_col_name='label', train_percent=20,
                                         window_size_for_metrics=window_size_for_metrics)
                    print_twitter_result(alpha, datapath, min_value, model, result, threshold)

                    if use_comet:
                        params = {'max_anoms': max_anoms,
                                  'threshold': threshold,
                                  'alpha': alpha,
                                  'window_size_for_metrics': window_size_for_metrics,
                                  'min_value': min_value
                                  }
                        metrics = result[model.__name__]
                        metrics = summarize_metrics(metrics)
                        log_experiment(datapath, dataset, model, parameters=params, metrics=metrics)

    if 'azure' in models_to_run or len(models_to_run) == 0:
        print("Evaluating Azure model")
        # Get Azure subscription id for the Azure Anomaly Detector
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../config/config.json')
        subscription_key = get_azure_subscription_key(filename)

        model = AzureAnomalyTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                               subscription_key=subscription_key, sensitivity=None)

        # Call Azure Anomaly Detector API
        prediction = model.predict(dataset, verbose=True)

        sensitivity = [None, 0.3, 0.5, 1, 1.5, 2]
        for sens in sensitivity:
            print("sensitivity = {}".format(sens))
            prediction = model.tune_prediction(prediction, sens)
            raw_metrics = get_metrics_for_all_categories(X[['value']], prediction[['prediction']], y[['label']],
                                                         window_size_for_metrics=5)
            metrics = get_final_metrics(raw_metrics)
            result = {}
            result[model.__name__] = metrics
            print_azure_model(datapath, min_value, model, result, sens)
            # model.plot(labels = y.reset_index().set_index('date'))

            if use_comet:
                params = {'sensitivity': sens,
                          'window_size_for_metrics': window_size_for_metrics
                          }
                metrics = result[model.__name__]
                metrics = summarize_metrics(metrics)
                log_experiment(datapath, dataset, model, parameters=params, metrics=metrics)


def print_azure_model(datapath, min_value, model, result, sens):
    print("Evaluating Azure")
    print('sens = ' + str(sens) +
          ' min_value = ' + str(min_value) +
          ', dataset = ' + datapath)
    print('F1 score = ' + str(result[model.__name__]['f1']) +
          ", precision = " + str(result[model.__name__]['precision']) +
          ", recall = " + str(result[model.__name__]['recall']))


def print_lstm_model(datapath, min_value, model, result):
    print("Evaluating LSTM")
    print('min_value = ' + str(min_value) +
          ', dataset = ' + datapath)
    print('F1 score = ' + str(result[model.__name__]['f1']) +
          ", precision = " + str(result[model.__name__]['precision']) +
          ", recall = " + str(result[model.__name__]['recall']))


def print_twitter_result(alpha, datapath, min_value, model, result, threshold):
    print("Evaluating Twitter")
    if threshold is None:
        print('Threshold = None, Alpha = ' + str(alpha) + ', max_anoms = None, min_value = ' + str(
            min_value) + ', dataset = ' + datapath)
    else:
        print('Threshold = ' + threshold + ', Alpha = ' + str(
            alpha) + ', max_anoms = None, min_value = ' + str(min_value) + ', dataset = ' + datapath)
    print('F1 score = ' + str(result[model.__name__]['f1']) + ", precision = " + str(
        result[model.__name__]['precision']) + ", recall = " + str(
        result[model.__name__]['recall']))


def print_stl_result(anomaly_type, datapath, min_value, model, num_std, result):
    print("Evaluating STL")
    print('num_std = ' + str(num_std) + ', anomaly_type = ' + str(anomaly_type) + ', min_value = ' + str(
        min_value) + ', dataset = ' + datapath)
    print('F1 score = ' + str(result[model.__name__]['f1']) + ", precision = " + str(result[model.__name__][
                                                                                         'precision']) + ", recall = " + str(
        result[model.__name__]['recall']))


def print_ma_result(anomaly_type, datapath, min_value, model, num_std, result):
    print("Evaluating MA")
    print('num_std = ' + str(num_std) + ', anomaly_type = ' + str(anomaly_type) + ', min_value = ' + str(
        min_value) + ', dataset = ' + datapath)
    print('F1 score = ' + str(result[model.__name__]['f1']) + ", precision = " + str(result[model.__name__][
                                                                                         'precision']) + ", recall = " + str(
        result[model.__name__]['recall']))


def log_experiment(datapath, dataset, model, parameters, metrics):
    experiment = Experiment(api_key="Uv0lx3yRDH7kk8h1vtR9ZRiD2s16gnYTxfsvK2VnpV2xRrMbFobYDZRRA4tvoYiR",
                            project_name="trending-topics")

    experiment.log_dataset_hash(dataset)
    experiment.log_parameter("model", model.__name__)
    experiment.log_parameter("dataset", datapath)

    for key, value in parameters.items():
        experiment.log_parameter(key, value)

    for key, value in metrics.items():
        if ~np.isnan(value):
            experiment.log_metric(key, value)


def get_azure_subscription_key(file):
    try:
        with open(file) as f:
            data = json.load(f)
            return data['subscription_key']
    except Exception as e:
        raise Exception("Error loading Azure subscription key for Azure Anomaly Finder.\n"
                        "Please create a json file and put your subscription_key value in it.\n"
                        "See https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/apps-anomaly-detection-api\n" + str(
            e))


if __name__ == '__main__':

    freqs = {'1': '30min', '2': '1H', '3': '3H', '4': '12H', '5': '24H'}
    cities = {'1': 'Corona', '2': 'Pompano', '3': 'SF'}
    models = {'s': 'stl', 'm': 'ma_seasonal', 't': 'twitter', 'a': 'azure', 'l': 'lstm'}

    city = 0
    while city not in ['1', '2', '3']:
        city = input("Select city: Corona (1), Pompano (2), SF (3):")

    freq = 0
    while freq not in ['1', '2', '3', '4', '5']:
        freq = input("Select time frequency: 30min (1), 1H (2), 3H (3), 12H (4) or 24H (5): ")

    model = 0
    while model not in ['s', 'm', 't', 'a', 'l', 'all']:
        model = input(
            "Select model to run: stl (s), ma_seasonal (m), twitter (t), "
            "azure anomaly finder (a), lstm (l) or all (all): ")

    datapath = "../datasets/{0}{1}_labeled.csv".format(cities[city], freqs[freq])
    if model == 'all':
        print("Loading file {0}. Evaluating all models".format(datapath))
        evaluate_all_models(datapath=datapath, freq=freqs[freq])
    else:
        print("Loading file {0}. Evaluating models {1}".format(datapath, models[model]))
        evaluate_all_models(datapath=datapath, freq=freqs[freq], models_to_run=models[model])
