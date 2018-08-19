import json
import os

from comet_ml import Experiment

from moda.models.azure_anomaly_detection.azure_ad import AzureAnomalyTrendinessDetector
from moda.models.data_reader import read_data
from moda.evaluators.eval import get_evaluation_metrics, get_final_metrics, eval_models
from moda.models.ma_seasonal.ma_seasonal_model import MovingAverageSeasonalTrendinessDetector
from moda.models.stl.stl_model import STLTrendinessDetector
from moda.models.twitter.anomaly_detect_multicategory import TwitterAnomalyTrendinessDetector


def run_model(datapath, freq, min_date='01-01-2018', plot=True, model_name='stl', min_value=10):
    print("Loading file {0}, with frequency {1}. Model name = {2}".format(datapath, freq, model_name))
    dataset = read_data(datapath)
    dataset = dataset.rename(columns={'is_anomaly': 'label'})

    if len(dataset.index.levels) > 1:
        categories = dataset.index.levels[1]
        print("categories found = {}".format(categories))

    if model_name == 'twitter':
        model = TwitterAnomalyTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value, threshold=None,
                                                 max_anoms=0.49, seasonality_freq=7)

    if model_name == 'ma_seasonal':
        model = MovingAverageSeasonalTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                                        anomaly_type='or',
                                                        num_of_std=3)

    if model_name == 'stl':
        model = STLTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                      anomaly_type='or',
                                      num_of_std=2.5, lo_delta=0)

    if model_name == 'azure':
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'config/config.json')
        subscription_key = get_azure_subscription_key(filename)
        model = AzureAnomalyTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                               subscription_key=subscription_key)

    prediction = model.predict(dataset, verbose=True)
    raw_metrics = get_evaluation_metrics(dataset[['value']], prediction[['prediction']], dataset[['label']],
                                         window_size_for_metrics=5)
    metrics = get_final_metrics(raw_metrics)
    print(metrics)

    ## Plot each category
    _, file = os.path.split(datapath)
    print("Plotting...")
    model.plot(labels=dataset['label'], postfix=file)


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


def eval_model(datapath="SF3H_labeled.csv", min_date='01-01-2018', freq='3H', use_comet=True):
    dataset = read_data(datapath)
    dataset = dataset[~dataset.index.duplicated(keep='first')]
    dataset = dataset.rename(columns={'is_anomaly': 'label'})

    X = dataset[['value']]
    y = dataset[['label']]

    min_value = 15
    window_size_for_metrics = 3

    # MA model

    anomaly_types = ['residual', 'trend', 'and', 'or']
    anomaly_type = 'residual'
    for num_std in [2, 2.5, 3, 3.5, 4]:
        for anomaly_type in anomaly_types:
            model = MovingAverageSeasonalTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                                            anomaly_type=anomaly_type, num_of_std=num_std)
            result = eval_models(X, y, [model], label_col_name='label', train_percent=20,
                                 window_size_for_metrics=window_size_for_metrics)
            print('num_std = ' + str(num_std) + ', anomaly_type = ' + str(anomaly_type) + ', min_value = ' + str(
                min_value) + ', dataset = ' + datapath)
            print('F1 score = ' + str(result[model.__name__]['f1']) + ", precision = " + str(result[model.__name__][
                                                                                                 'precision']) + ", recall = " + str(
                result[model.__name__]['recall']))
            # model.plot(labels = y.reset_index().set_index('date'))

            if use_comet:
                params = {'anomaly_type': anomaly_type,
                          'num_std': num_std,
                          'window_size_for_metrics': window_size_for_metrics,
                          'min_value': min_value
                          }
                metrics = result[model.__name__]
                metrics.pop('raw', None)
                log_experiment(datapath, dataset, model, parameters=params, metrics=metrics)

    # STL model
    for num_std in [2, 2.5, 3, 3.5, 4]:
        for anomaly_type in anomaly_types:
            for lo_frac in [0,0.1,0.5,1]:
                model = STLTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                              anomaly_type=anomaly_type, num_of_std=num_std,lo_frac=lo_frac)
                result = eval_models(X, y, [model], label_col_name='label', train_percent=20,
                                     window_size_for_metrics=window_size_for_metrics)
                print('num_std = ' + str(num_std) + ', anomaly_type = ' + str(anomaly_type) + ', min_value = ' + str(
                    min_value) + ', dataset = ' + datapath)
                print('F1 score = ' + str(result[model.__name__]['f1']) + ", precision = " + str(result[model.__name__][
                                                                                                     'precision']) + ", recall = " + str(
                    result[model.__name__]['recall']))
                # model.plot(labels = y.reset_index().set_index('date'))

                if use_comet:
                    params = {'anomaly_type': anomaly_type,
                              'num_std': num_std,
                              'window_size_for_metrics': window_size_for_metrics,
                              'min_value': min_value,
                              'lo_frac':lo_frac
                              }
                    metrics = result[model.__name__]
                    metrics.pop('raw', None)
                    log_experiment(datapath, dataset, model, parameters=params, metrics=metrics)

    max_anoms_list = [0.05, 0.1]
    for max_anoms in max_anoms_list:
        for threshold in [None, 'med_max', 'p95', 'p99']:
            for alpha in [0.05, 0.1, 0.15]:
                model = TwitterAnomalyTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                                         threshold=threshold, max_anoms=max_anoms,
                                                         longterm=False, alpha=alpha, seasonality_freq=7)

                result = eval_models(X, y, [model], label_col_name='label', train_percent=20,
                                     window_size_for_metrics=window_size_for_metrics)
                if threshold is None:
                    print('Threshold = None, Alpha = ' + str(alpha) + ', max_anoms = None, min_value = ' + str(
                        min_value) + ', dataset = ' + datapath)
                else:
                    print('Threshold = ' + threshold + ', Alpha = ' + str(
                        alpha) + ', max_anoms = None, min_value = ' + str(min_value) + ', dataset = ' + datapath)
                print('F1 score = ' + str(result[model.__name__]['f1']) + ", precision = " + str(
                    result[model.__name__]['precision']) + ", recall = " + str(
                    result[model.__name__]['recall']))

                if use_comet:
                    params = {'max_anoms': max_anoms,
                              'threshold': threshold,
                              'alpha': alpha,
                              'window_size_for_metrics': window_size_for_metrics,
                              'min_value': min_value
                              }
                    metrics = result[model.__name__]
                    metrics.pop('raw', None)
                    log_experiment(datapath, dataset, model, parameters=params, metrics=metrics)

    # Get Azure subscription id for the Azure Anomaly Detector
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'config/config.json')
    subscription_key = get_azure_subscription_key(filename)

    model = AzureAnomalyTrendinessDetector(is_multicategory=True, freq=freq, min_value=min_value,
                                           subscription_key=subscription_key, sensitivity=None)

    # Call Azure Anomaly Detector API
    prediction = model.predict(dataset, verbose=True)

    sensitivity = [None, 0.3, 0.5, 1, 1.5, 2]
    for sens in sensitivity:
        print("sensitivity = {}".format(sens))
        prediction = model.tune_prediction(prediction,sens)
        raw_metrics = get_evaluation_metrics(X[['value']], prediction[['prediction']], y[['label']],
                                             window_size_for_metrics=5)
        metrics = get_final_metrics(raw_metrics)
        result = {}
        result[model.__name__] = metrics
        print('sens = ' + str(sens) +
              ' min_value = ' + str(min_value) +
              ', dataset = ' + datapath)
        print('F1 score = ' + str(result[model.__name__]['f1']) +
              ", precision = " + str(result[model.__name__]['precision']) +
              ", recall = " + str(result[model.__name__]['recall']))
        # model.plot(labels = y.reset_index().set_index('date'))

        if use_comet:
            params = {'sensitivity': sens,
                      'window_size_for_metrics': window_size_for_metrics
                      }
            metrics = result[model.__name__]
            metrics.pop('raw', None)
            log_experiment(datapath, dataset, model, parameters=params, metrics=metrics)


def log_experiment(datapath, dataset, model, parameters, metrics):
    experiment = Experiment(api_key="Uv0lx3yRDH7kk8h1vtR9ZRiD2s16gnYTxfsvK2VnpV2xRrMbFobYDZRRA4tvoYiR",
                            project_name="trending-topics")

    experiment.log_dataset_hash(dataset)
    experiment.log_parameter("model", model.__name__)
    experiment.log_parameter("dataset", datapath)

    for key, value in parameters.items():
        experiment.log_parameter(key, value)

    for key, value in metrics.items():
        experiment.log_metric(key, value)


if __name__ == '__main__':

    inp1 = ''
    while inp1 not in ['r', 'e']:
        inp1 = input("Run (r) or evaluate (e)?")

    city = 0
    while city not in ['1', '2', '3', '9']:
        city = input("Select city: Corona (1), Pompano (2), SF (3), all (9):")

    paths = {#'1': "datasets/corona_labeled.csv",
             '2': "datasets/pompano_labeled.csv",
             '3': "datasets/SF30min_labeled.csv"}

    freqs = {'1': '12H', '2': '24H', '3': '30min'}

    models = {'s': 'stl', 'm': 'ma_seasonal', 't': 'twitter', 'a': 'azure'}

    if inp1 == 'r':
        model = input("Select model: Moving Averages (m), STL (s), Twitter (t), Azure Anomaly Detector (a):")
        run_model(datapath=paths[city], freq=freqs[city], model_name=models[model])
    if inp1 == 'e':
        if city == '9':
            print("Evaluating all cities")
            # eval_model(datapath="../test/dummy3.txt", freq='12H',use_comet=False)
            for val, freq in zip(paths, freqs):
                eval_model(datapath=paths[val], freq=freqs[freq], use_comet=True)
        else:
            print("Loading file {0}, with frequency {1}. Evaluating all models".format(paths[city], freqs[city]))
            eval_model(datapath=paths[city], freq=freqs[city])
