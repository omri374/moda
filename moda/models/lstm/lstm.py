"""
Contributions from:
DSEverything - Mean Mix - Math, Geo, Harmonic (LB 0.493)
https://www.kaggle.com/dongxu027/mean-mix-math-geo-harmonic-lb-0-493
JdPaletto - Surprised Yet? - Part2 - (LB: 0.503)
https://www.kaggle.com/jdpaletto/surprised-yet-part2-lb-0-503
hklee - weighted mean comparisons, LB 0.497, 1ST
https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st

Also all comments for changes, encouragement, and forked scripts rock
I've added my own model and used the prepared data by other awesoem kagglers!

Keep the Surprise Going
"""

import numpy as np
import pandas as pd
from keras.layers import TimeDistributed
from keras.losses import mean_squared_error

from moda.evaluators.eval import _prep_set
from moda.evaluators.metrics.metrics import _initialize_metrics
from moda.models import read_data


def lstm(X, y):
    from keras.layers import LSTM, Dense
    from keras.models import Sequential
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from math import floor

    datetimeindex = X.index.levels[0]
    num_dates = len(datetimeindex)
    train_percent = 70

    train = datetimeindex[:int(num_dates * train_percent / 100)]
    test = datetimeindex[int(num_dates * train_percent / 100):]
    categories = X.index.levels[1]

    counter = 0
    metrics = _initialize_metrics(categories)

    x_train = _prep_set(X, train)
    y_train = _prep_set(y, train)

    x_test = _prep_set(X, test)
    y_test = _prep_set(y, test)

    # Define the scaler
    scaler = StandardScaler().fit(x_train)
    # scaler = MinMaxScaler().fit(x_train)

    # Scale the train set
    x_train = scaler.transform(x_train)

    # Scale the test set
    x_test = scaler.transform(x_test)

    # Set random seed
    np.random.seed(7)

    print("--- shape report ---")
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_test: ", x_test.shape)

    # split the training for validation
    rate = 1.0
    train_sample_size = floor(x_train.shape[0] * rate)
    # commented out the validation
    # y_valid = np.copy(x_train[train_sample_size:,:])
    # y_valid = np.copy(y_train[train_sample_size:])
    x_train = x_train[:train_sample_size, :]
    y_train = y_train[:train_sample_size]

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    # x_valid = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    print("-- network input --")
    print("X_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    # print("X_valid: ", x_valid.shape)
    # print("y_valid: ", y_valid.shape)
    print("X_test: ", x_test.shape)

    # design network
    model = Sequential()
    model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(TimeDistributed())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # fit network
    # history = model.fit(x_train, y_train, epochs=10, batch_size=1000, \
    # validation_data=(x_valid, y_valid), verbose=2, shuffle=False)
    history = model.fit(x_train, y_train, epochs=60, batch_size=50, verbose=2, shuffle=False)

    # plot history
    plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # make a prediction for test data
    yhat = model.predict(x_test)
    # yhat = model.predict(x_valid)
    # yhat = model.predict(x_train)
    visitors = np.absolute(np.expm1(yhat))
    x_test['visitors'] = visitors
    #x_test[['id', 'visitors']].to_csv('submission_10.csv', index=False, float_format='%.3f')

    # calculate RMSE for the validation set
    # rmse = np.sqrt(mean_squared_error(yhat, y_valid))
    rmse = np.sqrt(mean_squared_error(yhat, y_train))
    print('Test RMSE: %.3f' % rmse)


if __name__ == '__main__':
    datapath = "../../datasets/corona_labeled.csv"
    df = read_data(datapath, min_date="01-01-2018")
    df = df.rename(columns={'is_anomaly': 'label'})

    one_category = df.loc[pd.IndexSlice[:, 'housing'], :]#.reset_index(level='category', drop=True)

    X = one_category[['value']]
    y = one_category[['label']]

    lstm(X, y)
