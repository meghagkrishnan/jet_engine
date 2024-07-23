from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

import time

#from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
#print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
print("\n✅ Loading Tensorflow...")
start = time.perf_counter()

from tensorflow import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, Normalization, LSTM
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def train_base_model(X_train, y_train):
    #Create a base model
    base_model = LinearRegression()

    # Train the model using the training sets
    base_model.fit(X_train, y_train)

    return base_model

def LR_model_evaluate(y_test, y_pred):

    R2_score = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("\n✅ Baseline model evaluation done")

    return R2_score, rmse

def train_LSTM_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=16, patience=5):
    """
    Train an RNN model (LSTM) for predictive maintenance.
    Parameters:
    X_train (np.array): Training features in (n_seq, n_obs, n_features) format.
    y_train (np.array): Training labels.
    X_val (np.array): Validation features in (n_seq, n_obs, n_features) format.
    y_val (np.array): Validation labels.
    seq_length (int): Length of each sequence.
    n_features (int): Number of features in each sequence.
    epochs (int): Number of epochs for training.
    batch_size (int): Batch size for training.
    patience (int): Number of epochs to wait for improvement before stopping early.
    Returns:
    model: Trained Keras model.
    history: Training history.
    """
    normalizer = Normalization()
    normalizer.adapt(X_train)

    # Define the model
    model = Sequential()
    model.add(normalizer)
    model.add(LSTM(units=100, activation='tanh',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='tanh', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mse',
                  optimizer='adam',
                 metrics=['mae'])

    # Define EarlyStopping callback
    es = EarlyStopping(patience=patience, restore_best_weights=True, monitor='val_loss')

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )
    print("\n✅ LSTM model training done")
    return model, history

def LSTM_model_evaluate(y_test, y_pred):

    #mae = model_LSTM.evaluate(X_test, y_test, verbose=0)
    #print(f'Model Mean Absolute Error {LSTM_mae[1]:.4f}')

    R2_score = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("\n✅ LSTM model evaluation done")

    return R2_score, rmse
