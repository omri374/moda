# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
from moda.example.example import prep_data

experiment = Experiment(api_key="Uv0lx3yRDH7kk8h1vtR9ZRiD2s16gnYTxfsvK2VnpV2xRrMbFobYDZRRA4tvoYiR",
                        project_name="keras-lstm-sf311-forecast", workspace="omri374")

import pandas as pd

from moda.dataprep.create_dataset import get_windowed_ts, split_history_and_current
from moda.models import read_data


def lstm_forecast(window_size):
    from keras.layers.recurrent import LSTM
    from keras.models import Sequential
    from keras.layers import Conv1D
    from keras.layers.core import Dense, Activation

    model = Sequential()
    model.add(Conv1D(input_shape=(window_size, 1), filters=32, kernel_size=10))
    model.add(LSTM(output_dim=window_size, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam")
    print(model.summary())

    return model


def scale(X):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaled_values = scaler.fit_transform(X['value'].values.reshape(-1, 1))
    scaled = pd.DataFrame(index=X.index)
    scaled['value'] = scaled_values

    return scaled


if __name__ == '__main__':
    datapath = "../../example/SF311_simplified.csv"
    df = prep_data(datapath, min_date="01-01-2018")
    #df = df.rename(columns={'is_anomaly': 'label'})

    one_category = df.loc[pd.IndexSlice[:, 'Street and Sidewalk Cleaning'], :].reset_index(level='category', drop=True)
    one_category_scaled = scale(one_category)

    train_percent = 70
    datetimeindex = one_category_scaled.index
    num_dates = len(datetimeindex)

    train = one_category_scaled.loc[datetimeindex[:int(num_dates * train_percent / 100)]]
    test = one_category_scaled.loc[datetimeindex[int(num_dates * train_percent / 100):]]

    print("Training set length = {0}, Test set length = {1}".format(len(train), len(test)))

    as_windows_train = get_windowed_ts(train, window_size=20)
    as_windows_test = get_windowed_ts(test, window_size=20)
    train_X, train_y = split_history_and_current(as_windows_train)
    test_X, test_y = split_history_and_current(as_windows_test)

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
    model = lstm_forecast(window_size=20)

    import time

    start = time.time()
    model.fit(train_X, train_y, batch_size=256, epochs=3, validation_split=0.1)
    print("> Compilation Time : ", time.time() - start)

    # Doing a prediction on all the test data at once
    preds = model.predict(test_X)

    from sklearn.metrics import mean_squared_error

    actual = test_y
    prediction = preds
    print("MSE=" + str(mean_squared_error(actual, prediction)))
