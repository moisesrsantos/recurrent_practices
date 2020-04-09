import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from numpy import array
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from rnn import *


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


df = pd.read_csv("./data/fa_increasing_trend.data", header=None)

raw_seq = df.values
plt.figure(1)
plt.plot(raw_seq)
plt.show()

n_steps = 5
X, y = split_sequence(raw_seq, n_steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

n_features = 1
hidden = 50
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))

X_train, model = vanilla_lstm(X_train, hidden, n_steps, n_features)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, decay = 1e-4), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=200,
                    batch_size=5,
                    validation_split=0.1,
                    verbose=1,
                    shuffle=False)

x_input = X_test
yhat = model.predict(x_input, verbose=0)

plt.figure(2)
plt.plot(y_test)
plt.plot(yhat)
plt.show()

print("MAE: " + str(mean_absolute_error(yhat, y_test)))

# Plot training & validation loss values
plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
