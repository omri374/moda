from keras.utils import np_utils

if __name__ == '__main__':
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, TimeDistributed
    from keras.layers import LSTM

    # prepare sequence
    length = 10
    #    seq = array([i / float(length) for i in range(length)])
    X = np.repeat(np.array([0, 0.9, 0, 0, 0, 0.8, 0, 0, 0, 0.1]),1000)
    y = np.repeat(np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),1000)
    X = X.reshape(1, len(X), 1)
    y = np_utils.to_categorical(y)
    # define LSTM configuration
    n_neurons = length
    n_batch = length
    n_epoch = 1000
    # create LSTM
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(length, 1)))
    model.add(TimeDistributed(Dense(2)))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print(model.summary())
    # train LSTM
    model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
    # evaluate
    result = model.predict(X, batch_size=n_batch, verbose=0)
    for value in result:
        print('%.1f' % value)
