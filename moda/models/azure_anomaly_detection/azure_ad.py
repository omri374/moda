import json

import numpy as np
import pandas as pd
import requests

from moda.models.trend_detector import AbstractTrendDetector, MIN_SAMPLES_PER_CATEGORY

ENDPOINT = "https://api.labs.cognitive.microsoft.com/anomalyfinder/v1.0/anomalydetection"


class AzureAnomalyTrendinessDetector(AbstractTrendDetector):
    GENERAL_CATEGORY = 'general'

    __name__ = 'AzureAnomalyTrendinessDetector'

    def __init__(self, freq, subscription_key, is_multicategory=False, min_value=None, resample=True, sensitivity=1.0,
                 endpoint=ENDPOINT):
        super(AzureAnomalyTrendinessDetector, self).__init__(freq, is_multicategory, resample)
        self.min_value = min_value
        self.sensitivity = sensitivity
        self.endpoint = endpoint
        self.subscription_key = subscription_key
        self.results = None

    def fit_one_category(self, dataset, category=None, verbose=False):
        count_list = dataset['value'].values
        dates_list = dataset.index.tolist()
        points = [{"Timestamp": date.strftime(format='%Y-%m-%dT%H:%M:%S'), "Value": count} for date, count in
                  zip(dates_list, count_list)]

        request_data = {}
        request_data["Points"] = points

        def detect(url, subscription_key, request_data):
            headers = {'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key': subscription_key}
            response = requests.post(url, data=json.dumps(request_data), headers=headers)
            if response.status_code == 200:
                return json.loads(response.content.decode("utf-8"))
            else:
                print(response.status_code)
                raise Exception(response.text)

        model_response = detect(self.endpoint, self.subscription_key, request_data)

        results = pd.DataFrame({'expected': model_response['ExpectedValue'], 'is_anomaly': model_response['IsAnomaly'],
                                'is_anomaly_neg': model_response['IsAnomaly_Neg'],
                                'is_anomaly_pos': model_response['IsAnomaly_Pos'],
                                'upper': model_response['UpperMargin'],
                                'lower': model_response['LowerMargin'], 'date': dates_list})

        results = pd.merge(dataset, results, how='inner', on='date')

        values = [x['Value'] for x in request_data['Points']]
        anomalies = []
        for (index, value) in results['is_anomaly'].iteritems():
            if value > 0 and (
                values[index] > results.iloc[index]['expected'] + self.sensitivity * results.iloc[index]['upper'] or
                values[index] < results.iloc[index]['expected'] - self.sensitivity * results.iloc[index]['lower']):
                anomalies.append(1)
            else:
                anomalies.append(0)

        results['prediction'] = anomalies

        if self.min_value is not None:
            results['prediction'] = np.where(results['value'] < self.min_value, 0,
                                             results['prediction'])

        self.input_data[category] = results

        return results

    def predict_one_category(self, X, category):
        results = self.input_data.get(category)

        if results is None:
            test = X
        else:
            test = pd.concat([results, X], sort=True)
            test = test[~test.index.duplicated(keep='first')]

        ## We have to fit the entire datasets again (train+test) as we can't compute anomalies iteratively
        if len(test) > MIN_SAMPLES_PER_CATEGORY:
            test = self.fit_one_category(test, category)

        return test

    def plot_one_category(self, category=None, labels=None):
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter

        if self.input_data is None:
            print("Empty datasets")
            return None

        if len(self.input_data) == 0:
            print("Empty datasets")
            return None

        if category not in self.input_data.keys():
            print("Empty datasets")

        fig, ax = plt.subplots(figsize=(20, 10))

        dataset = self.input_data[category]
        dates = dataset['date']
        values = dataset['value']
        prediction = dataset['prediction']
        upper = dataset['upper']
        lower = dataset['lower']
        expected = dataset['expected']

        myFmt = DateFormatter('%Y-%m-%d')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(myFmt)

        ## Rotate date labels automatically
        fig.autofmt_xdate()
        ax.plot(dates, values, label='Value', linestyle="-")
        if ~np.all(prediction == 0):
            mask = (~np.isnan(prediction)) & (prediction != 0)
            ax.plot(dates[mask], values[mask], marker='o', color='r', label='Prediction')
            # ax.plot(dates, values, marker='o', color='r')
        ax.plot(dates, expected, label='Expected', linestyle=':')
        ax.plot(dates, expected + self.sensitivity * upper, label='Upper', linestyle='-.',
                linewidth=1)
        ax.plot(dates, expected - self.sensitivity * lower, label='Lower', linestyle='-.',
                linewidth=1)

        if (labels is None) and ('label' in self.input_data[category]):
            labels = self.input_data[category]['label']

        if labels is not None:
            cols_to_use = dataset.columns.difference(labels.columns)
            labels_padded = pd.merge(dataset[cols_to_use], labels, how='left', on='date')
            mask = (~np.isnan(labels_padded['label'])) & (labels_padded['label'] != 0)
            true_labels = labels_padded.loc[mask]
            if len(true_labels) > 0:
                ax.plot(true_labels['date'], true_labels['value'], label='Labels', marker='x',
                        color='b')

        if category is None:
            plt.title("Azure anomaly detection results for sensitivity = " + str(
                self.sensitivity))
        else:
            plt.title("Azure anomaly detection results for category = " + category + ", sensitivity = " + str(
                self.sensitivity))
        ax.legend(shadow=True, fancybox=True)
