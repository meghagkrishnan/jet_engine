import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def seperate_data(df):
    X = df.drop(columns = ['RUL'])
    y = df['RUL']
    return X, y

def process_Xdata(X):
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(features_normalized, columns=X.columns)

    return X_scaled

def split_data_LR(X, y):

    X_train = X[X['id'] <= 80]
    X_test = X[X['id'] > 80]
    y_train = y[:len(X_train)]
    y_test = y[len(X_train):]

    return X_train, X_test, y_train, y_test


def split_data_RNN(data):

    data_train = data[data['id'] <= 80]
    data_test = data[data['id'] > 80]

    print(f'Train data shape = {data_train.shape}, Test data shape = {data_test.shape}')

    return data_train, data_test

def data_preperation_RNN(data, seq_length=30):
    #This function create a sequence of the data in (n_seq, n_obs, n_features) format to train DL methods
    sequences = []
    labels = []
    for unit in data['id'].unique():
        unit_data = data[data['id'] == unit].sort_values(by='cycle')
        num_sequences = len(unit_data) - seq_length + 1
        for i in range(num_sequences):
            seq = unit_data.iloc[i: i+seq_length]
            sequences.append(seq.drop(columns=['id', 'cycle', 'RUL']).values)
            labels.append(seq['RUL'].values[-1])
    X = np.array(sequences)
    y = np.array(labels)
    y = np.expand_dims(y, axis=1)
    return X, y

def val_split_RNN(data, val_size = 0.3):

    val_split = int(data.shape[0]*val_size)
    data_train = data[val_split:]
    data_val = data[:val_split]

    return data_train, data_val
