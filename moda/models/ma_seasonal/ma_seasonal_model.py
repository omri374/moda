import traceback

import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from moda.models.trend_detector import *


class MovingAverageSeasonalTrendinessDetector(AbstractTrendDetector):
    __name__ = 'MovingAverageSeasonalTrendinessDetector'
    """An detector for anomalies on time series using the seasonal_decompose method

    Parameters
    ----------
    is_multicategory : boolean
        Whether the provided dataset contains multiple categories per time stamp (True), or just one value per timestamp (False)
    num_of_std : float
        Sets the threshold for anomalies using the median + std*num_of_std rule, either for trends or residuals
    freq : String
        The time-series frequency, used to fill missing date/times. See https://pandas.pydata.org/pandas-docs/stable/timeseries.html

    seasonality : integer
        The seasonality of the data. See http://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html

    anomaly_type: String
        The values on which anomalies are looked for. Possible values are 'trend','residual' and a list of both
    """

    GENERAL_CATEGORY = 'general'

    def __init__(self, freq, is_multicategory=True, num_of_std=3, seasonality_freq=7, min_value=None,
                 anomaly_type=['trend', 'residual', 'and', 'or'], resample=True, min_periods=10):
        super(MovingAverageSeasonalTrendinessDetector, self).__init__(freq, is_multicategory, resample)
        self.num_of_std = num_of_std
        self.min_value = min_value
        self.seasonality = seasonality_freq
        self.anomaly_type = anomaly_type
        self.min_periods = min_periods

    def fit_one_category(self, dataset, category=None, verbose=False):
        """Returns anomalies of the trend based on a k*sd heuristic

         Parameters
         ----------
         dataset: pandas.DataFrame
             A data frame containing a date/time index and 'value' column

         Returns
         -------
         df : pandas.DataFrame
             Returns a pandas DataFrame with a date/time index, with these columns:
                The seasonality decomposition:
                - trend
                - seasonality
                - residual
                A bool containing True if a trend anomaly was detected and False otherwise.
                Note that the returned data frame fills the values of missing dates with 0.
         """

        ts = dataset['value']
        # labels = datasets['is_anomaly']
        results = pd.DataFrame(index=ts.index)
        results['value'] = ts
        # results['labels'] = labels
        results['prediction'] = None

        if len(dataset) < MIN_SAMPLES_PER_CATEGORY:
            if self.input_data is None:
                self.input_data = {}

            self.input_data[category] = results
            return results

        # print(ts.index.freq)
        try:
            decomposition = seasonal_decompose(ts.values, freq=self.seasonality)
        except ValueError as e:
            if verbose:
                print(
                    "Error running stl due to  " + str(
                        e) + ". Returning 0 for all predictions\n" + traceback.format_exc())
            return results

        results = pd.DataFrame(index=ts.index)

        if 'labels' in dataset:
            results['labels'] = dataset['labels']

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        results['value'] = ts
        results['trend'] = trend
        results['residual'] = residual
        results['seasonality'] = seasonal

        results['residual_median'] = results['residual'].rolling('30D', min_periods=self.min_periods).median()
        results['residual_std'] = results['residual'].rolling('30D', min_periods=self.min_periods).std()

        results['trend_median'] = results['trend'].rolling('30D', min_periods=self.min_periods).median()
        results['trend_std'] = results['trend'].rolling('30D', min_periods=self.min_periods).std()

        results['residual_anomaly'] = np.where(
            residual > (results['residual_median'] + (results['residual_std'] * self.num_of_std)), 1, 0)
        results['trend_anomaly'] = np.where(trend > results['trend_median'] + (results['trend_std'] * self.num_of_std),
                                            1, 0)

        if self.anomaly_type == 'or':
            results['prediction'] = (results['residual_anomaly'].values == 1) | (results['trend_anomaly'].values == 1)
        elif self.anomaly_type == 'trend':
            results['prediction'] = results['trend_anomaly']
        elif self.anomaly_type == 'residual':
            results['prediction'] = results['residual_anomaly']
        elif self.anomaly_type == 'and':
            results['prediction'] = (results['residual_anomaly'].values == 1) & (results['trend_anomaly'].values == 1)
        if self.input_data is None:
            self.input_data = {}

        ## Remove predictions of values less than threshold
        if self.min_value is not None:
            results['prediction'] = np.where(results['value'] < self.min_value, 0, results['prediction'])

        self.input_data[category] = results
        return results

    def predict_one_category(self, X, category):
        results = self.input_data.get(category)

        if results is None:
            test = X
        else:
            test = pd.concat([results, X], sort=True)
            test = test[~test.index.duplicated(keep='first')]
            test = test.resample(self.freq, convention='start').asfreq().fillna(0)

        ## We have to fit the entire datasets again (train+test) as we can't compute trend and residuals otherwise.
        test = self.fit_one_category(test, category)
        if len(test) < MIN_SAMPLES_PER_CATEGORY:
            return test

        ## Remove predictions of values less than threshold
        if self.min_value is not None:
            test['prediction'] = np.where(test['value'] < self.min_value, 0, test['prediction'])

        test = pd.merge(test, X, on=['date', 'value'])

        return test

    def plot_one_category(self, category=None, labels=None):
        import matplotlib.pyplot as plt

        if self.input_data is None:
            print("Empty datasets")
            return None

        if len(self.input_data) == 0:
            print("Empty datasets")
            return None

        if category not in self.input_data.keys():
            print("Empty datasets")

        def ts_subplot(plt, series, label):
            plt.plot(series, label=label, linewidth=0.5)
            plt.legend(loc='best')
            plt.xticks(rotation=90)

        plt.subplot(421, )
        ts_subplot(plt, self.input_data[category]['value'], label='Original')
        plt.subplot(422)
        ts_subplot(plt, self.input_data[category]['residual_anomaly'], label='Residual anomaly')
        plt.subplot(423)
        ts_subplot(plt, self.input_data[category]['trend'], label='Trend (stl)')
        plt.subplot(424)
        ts_subplot(plt, self.input_data[category]['trend_anomaly'], label='Trend anomaly')
        plt.subplot(425)
        ts_subplot(plt, self.input_data[category]['seasonality'], label='Seasonality')
        plt.subplot(426)
        ts_subplot(plt, self.input_data[category]['prediction'], label='Prediction')
        plt.subplot(427)
        ts_subplot(plt, self.input_data[category]['residual'], label='Residual')

        if 'labels' in self.input_data[category]:
            plt.subplot(428)
            ts_subplot(plt, self.input_data[category]['labels'], label='Labels')
        elif labels is not None:
            plt.subplot(428)
            ts_subplot(plt, labels, label='Labels')

        if category is None:
            plt.suptitle("STL results for threshold (std)=" + str(
                self.num_of_std) + ", seasonality_freq=" + str(self.seasonality) + ", freq=" + str(self.freq))
        else:
            plt.suptitle("STL results for category " + category + ", threshold (std)=" + str(
                self.num_of_std) + ", seasonality_freq=" + str(self.seasonality) + ", freq=" + str(self.freq))
