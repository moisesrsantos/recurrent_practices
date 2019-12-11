from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D



def vanilla_lstm(X, hidden, n_steps, n_features):
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(LSTM(hidden, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return X, model

def stacked_lstm(X, hidden, n_steps, n_features):
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(LSTM(hidden, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(hidden, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return X, model

def bidirectional_lstm(X, hidden, n_steps, n_features):
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return X, model

def cnn_lstm(X, hidden, n_steps, n_features, n_seq):
    X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(hidden, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return X, model


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

n_steps = 4
X, y = split_sequence(raw_seq, n_steps)
n_features = 1
n_seq = 2
n_steps = 2
hidden = 50

#X, model = bidirectional_lstm(X, hidden, n_steps, n_features)
X, model = cnn_lstm(X, hidden, n_steps, n_features, n_seq)

model.fit(X, y, epochs=500, verbose=0)

x_input = array([60, 70, 80, 90])
#x_input = x_input.reshape((1, n_steps, n_features))
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
