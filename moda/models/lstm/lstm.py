## Notice: this is still WIP and not working as a part of the moda framework

# import comet_ml in the top of your file
import time

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Activation

from moda.dataprep.create_dataset import get_windowed_ts, split_history_and_current
from moda.models.trend_detector import AbstractTrendDetector
import pandas as pd
import numpy as np




class LSTMTrendinessDetector(AbstractTrendDetector):
    __name__ = 'LSTMTrendinessDetector'
    """An detector for anomalies on time series using an LSTM time series forecasting model

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
        The values on which anomalies are looked for. Possible values are ['trend', 'residual', 'and', 'or']

    resample : bool
        Whether to add missing dates/times to time series prior to modeling

    min_periods : numeric
        The minimum value of periods for which to attemp to model. A time series shorter than that will return prediction = 0 

    lookback : String
        The time for which statistics for a specific timestamp look back. i.e. A sliding window for statistics (median, std). Example: '30D','12H' etc.
    """

    def __init__(self, freq, is_multicategory=True, num_of_std=3, min_value=None,window_size = 24,batch_size = 256, epochs = 10,
                 resample=True, min_periods=10, lookback='30D'):
        super(LSTMTrendinessDetector, self).__init__(freq, is_multicategory, resample)
        self.lookback = lookback
        self.num_of_std = num_of_std
        self.min_value = min_value
        self.min_periods = min_periods
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = {}

    def lstm_forecast_model(self, window_size,verbose=False):
        model = Sequential()
        # model.add(Conv1D(input_shape=(window_size, 1), filters=32, kernel_size=10))
        # model.add(MaxPooling1D(pool_size=5))
        model.add(LSTM(input_shape=(window_size, 1), output_dim=window_size, return_sequences=True))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dense(1))
        model.add(Activation("linear"))
        model.compile(loss="mse", optimizer="adam")
        if verbose:
            print(model.summary())

        return model

    def fit_one_category(self, dataset, category=None, verbose=False):
        train = scale(dataset)

        as_windows_train = get_windowed_ts(train, window_size=self.window_size)
        train_X, train_y = split_history_and_current(as_windows_train)

        train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)

        # Get model definition
        model = self.lstm_forecast_model(window_size=self.window_size,verbose=verbose)


        # Fit model
        start = time.time()
        model.fit(train_X, train_y, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1,
                  callbacks=[EarlyStopping(patience=2)])
        print("> Compilation Time : ", time.time() - start)

        self.model[category] = model

    def predict_one_category(self, X, category):
        test = scale(X)

        as_windows_test = get_windowed_ts(test, window_size=self.window_size)
        test_X, test_y = split_history_and_current(as_windows_test)

        test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

        preds = self.model[category].predict(test_X)

        from sklearn.metrics import mean_squared_error

        actual = test_y
        prediction = preds
        mse = mean_squared_error(actual, prediction)
        print("MSE=" + str(mse))
        results = pd.DataFrame({"date": X.index, "prediction": preds.squeeze(), "actual": test_y, "diff": test_y - preds.squeeze()})
        results['diff_rolling_std'] = results['diff'].rolling(self.lookback, min_periods=self.min_periods).std()  # naive version where we look at the standard deviation of the difference between predicted and actual in a previous window.
        results['prediction'] = np.where(results['diff'] > self.num_of_std * results['diff_rolling_std'], 1, 0)
        return results


def scale(X):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaled_values = scaler.fit_transform(X['value'].values.reshape(-1, 1))
    scaled = pd.DataFrame(index=X.index)
    scaled['value'] = scaled_values

    return scaled
