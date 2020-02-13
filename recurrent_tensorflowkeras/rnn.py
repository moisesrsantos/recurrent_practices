from tensorflow import keras


def vanilla_lstm(X, hidden, n_steps, n_features):
    main_input = keras.Input(shape=(X.shape[1:]))
    lstm_out = keras.layers.LSTM(units=hidden, activation='relu', input_shape=(n_steps, n_features))(main_input)
    x = keras.layers.Dense(25, activation='relu')(lstm_out)
    x = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=[main_input], outputs=[x])
    return X, model

    # X = X.reshape((X.shape[0], X.shape[1], n_features))
    # model = keras.Sequential()
    # model.add(keras.layers.LSTM(units=hidden, activation='relu', input_shape=(n_steps, n_features)))
    # model.add(keras.layers.Dense(units=1))
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    # return X, model


def stacked_lstm(X, hidden, n_steps, n_features):
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = keras.Sequential()
    model.add(keras.layers.LSTM(hidden, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(keras.layers.LSTM(hidden, activation='relu'))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    return X, model


def bidirectional_lstm(X, hidden, n_steps, n_features):
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = keras.Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(hidden, activation='relu', return_sequences=True),
                                         input_shape=(n_steps, n_features)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=hidden, activation='relu')))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    return X, model


def hybrid_lstm(X, hidden, n_steps, n_features, X_aux):
    main_input = keras.Input(shape=(X.shape[1:]), name='main_input')
    lstm_out = keras.layers.LSTM(units=hidden, activation='relu', input_shape=(n_steps, n_features))(main_input)
    auxiliary_input = keras.Input(shape=(X_aux.shape[1:]), name='aux_input')
    x = keras.layers.concatenate([lstm_out, auxiliary_input])
    x = keras.layers.Dense(25, activation='relu')(x)
    main_output = keras.layers.Dense(1, activation='relu', name='main_output')(x)
    model = keras.Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    return X, model

    # model = keras.Sequential()
    # model.add(keras.layers.LSTM(units=hidden, activation='relu', input_shape=(n_steps, n_features)))
    # model.add(keras.layers.Dense(units=1))
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    # return X, model
