import numpy as np
import pandas as pd

from pathlib import Path

from jetengine.params import *
from jetengine.ml_logic.data import clean_Xdata, clean_train_data
from jetengine.ml_logic.registry import save_LSTM_model, save_LR_model, load_LSTM_model, load_LR_model
from jetengine.ml_logic.model import LR_model_evaluate, train_base_model, train_LSTM_model, LSTM_model_evaluate
from jetengine.ml_logic.preprocessor import seperate_data,process_Xdata, split_data_LR, split_data_RNN, data_preperation_RNN, val_split_RNN

data_query_cache_path = Path(LOCAL_DATA_PATH)
train_path = data_query_cache_path.joinpath("train_FD001.txt")
test_path = data_query_cache_path.joinpath("test_FD001.txt")
rul_path = data_query_cache_path.joinpath("RUL_FD001.txt")

if train_path.is_file():
    print("✅ Loading train data from local txt...")
    train_FD001 = pd.read_csv(train_path, sep = ' ', header = None)
    #test_FD001 = pd.read_csv(test_path, sep = ' ', header=None)
    #rul_FD001 = pd.read_csv(rul_path, sep = ' ', header=None)
else:
    print("❌ No data in the folder...")

train_data = clean_train_data(train_FD001)
'''
def baseline_model_score(train_data):

    X,y = seperate_data(train_data) #Seperated X and target
    #Split data into test and train set
    X_train, X_test, y_train, y_test = split_data_LR(X, y)


    X_train_cleaned = clean_Xdata(X_train)  #Dropped id and cycle columns
    X_test_cleaned = clean_Xdata(X_test)
    X_train_processed = process_Xdata(X_train_cleaned)  #Standard Scaler
    X_test_processed = process_Xdata(X_test_cleaned)
    baseline_model = train_base_model(X_train_processed, y_train)
    save_LG_model(baseline_model)

    y_pred = baseline_model.predict(X_test_processed)
    baseline_score = LR_model_evaluate(y_test, y_pred)

    return baseline_score
'''
def LSTM_model_score(train_data):
    data_train, data_test = split_data_RNN(train_data)
    seq_length = 50
    data_X_train, data_y_train = data_preperation_RNN(data_train, seq_length=seq_length)
    X_test, y_test = data_preperation_RNN(data_test, seq_length=seq_length)
    X_train, X_val = val_split_RNN(data_X_train, val_size = 0.3)
    y_train, y_val = val_split_RNN(data_y_train, val_size = 0.3)
    LSTM_model, LSTM_history = train_LSTM_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=16, patience=5)
    save_LSTM_model(LSTM_model)
    y_pred_LSTM = LSTM_model.predict(X_test)
    LSTM_scores = LSTM_model_evaluate(y_test, y_pred_LSTM)

    return LSTM_scores

def load_model_score_LR(train_data):
    X,y = seperate_data(train_data) #Seperated X and target
    #Split data into test and train set
    X_train, X_test, y_train, y_test = split_data_LR(X, y)


    X_train_cleaned = clean_Xdata(X_train)  #Dropped id and cycle columns
    X_test_cleaned = clean_Xdata(X_test)
    X_train_processed = process_Xdata(X_train_cleaned)  #Standard Scaler
    X_test_processed = process_Xdata(X_test_cleaned)
    baseline_model = load_LR_model()
    y_pred = baseline_model.predict(X_test_processed)
    baseline_score = LR_model_evaluate(y_test, y_pred)
    return baseline_score

def load_model_score_LSTM(train_data):
    data_train, data_test = split_data_RNN(train_data)
    seq_length = 50
    #data_X_train, data_y_train = data_preperation_RNN(data_train, seq_length=seq_length)
    X_test, y_test = data_preperation_RNN(data_test, seq_length=seq_length)
    print("✅")
    print(X_test.shape)
    LSTM_model = load_LSTM_model()
    y_pred_LSTM = LSTM_model.predict(X_test)
    LSTM_scores = LSTM_model_evaluate(y_test, y_pred_LSTM)
    return LSTM_scores

if __name__ == '__main__':
    try:
        print(f'✅ Baseline score: R2, RMSE = {load_model_score_LR(train_data)} ')
        print(f'✅ LSTM score: R2, RMSE = {load_model_score_LSTM(train_data)} ')
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
