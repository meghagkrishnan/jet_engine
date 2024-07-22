import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Split Data into Train and Validation Sets, split (80% training and 20% validation)

def split_data(df, train_ratio=0.8):
    unit_ids = df['id'].unique()
    train_ids = unit_ids[:int(train_ratio * len(unit_ids))]
    val_ids = unit_ids[int(train_ratio * len(unit_ids)):]

    train_data = df[df['id'].isin(train_ids)]
    val_data = df[df['id'].isin(val_ids)]

    return train_data, val_data

# Normalize the sensor measurements and operational settings using MinMaxScaler.

def normalize_data(train_data, val_data, test_data):
    scaler = MinMaxScaler()

    # Columns to be normalized (exclude 'unit_number', 'time_in_cycles', and 'RUL')
    feature_cols = [col for col in train_data.columns if col not in ['id', 'cycle', 'RUL']]

    # Fit the scaler on training data
    train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])

    # Apply the scaler on validation and test data
    val_data[feature_cols] = scaler.transform(val_data[feature_cols])
    test_data[feature_cols] = scaler.transform(test_data[feature_cols])

    return train_data, val_data, test_data

def generate_sequences(df, seq_length):
    """
    Generate sequences of the data in (n_seq, n_obs, n_features) format for DL methods.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    seq_length (int): Length of each sequence.

    Returns:
    np.array: Array of sequences with shape (n_seq, n_obs, n_features).
    """
    sequences = []
    for unit in df['id'].unique():
        unit_data = df[df['id'] == unit].sort_values(by='cycle')  # Sort by cycle
        num_sequences = len(unit_data) - seq_length + 1
        for i in range(num_sequences):
            seq = unit_data.iloc[i:i + seq_length]
            sequences.append(seq.drop(columns=['id', 'cycle', 'RUL']).values)
    return np.array(sequences)

def generate_labels(df, seq_length):
    """
    Generate labels for each sequence based on the last RUL value in the sequence.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    seq_length (int): Length of each sequence.

    Returns:
    np.array: Array of labels.
    """
    labels = []
    for unit in df['id'].unique():
        unit_data = df[df['id'] == unit].sort_values(by='cycle')  # Sort by cycle
        num_sequences = len(unit_data) - seq_length + 1
        for i in range(num_sequences):
            seq = unit_data.iloc[i:i + seq_length]
            labels.append(seq['RUL'].values[-1])  # Get the last RUL value in the sequence
    return np.array(labels)
