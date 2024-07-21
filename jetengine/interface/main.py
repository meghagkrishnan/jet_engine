import numpy as np
import pandas as pd

from pathlib import Path

from jetengine.params import *
from jetengine.ml_logic.data import clean_Xdata, clean_train_data
from jetengine.ml_logic.model import LR_model_evaluate, train_base_model
from jetengine.ml_logic.preprocessor import seperate_data,process_Xdata, split_data_LR

data_query_cache_path = Path(LOCAL_DATA_PATH)
train_path = data_query_cache_path.joinpath("train_FD001.txt")
test_path = data_query_cache_path.joinpath("test_FD001.txt")
rul_path = data_query_cache_path.joinpath("RUL_FD001.txt")

def baseline_model_score():
    if train_path.is_file():
        print("✅ Loading train data from local txt...")
        train_FD001 = pd.read_csv(train_path, sep = ' ', header = None)
        #test_FD001 = pd.read_csv(test_path, sep = ' ', header=None)
        #rul_FD001 = pd.read_csv(rul_path, sep = ' ', header=None)
    else:
        print("❌ No data in the folder...")

    train_data = clean_train_data(train_FD001)
    X,y = seperate_data(train_data) #Seperated X and target
    #Split data into test and train set
    X_train, X_test, y_train, y_test = split_data_LR(X, y)

    X_train_cleaned = clean_Xdata(X_train)  #Dropped id and cycle columns
    X_test_cleaned = clean_Xdata(X_test)
    X_train_processed = process_Xdata(X_train_cleaned)  #Standard Scaler
    X_test_processed = process_Xdata(X_test_cleaned)
    baseline_model = train_base_model(X_train_processed, y_train)

    y_pred = baseline_model.predict(X_test_processed)
    baseline_score = LR_model_evaluate(y_test, y_pred)

    return baseline_score


if __name__ == '__main__':
    try:
        print(f'Baseline score: R2, RMSE = {baseline_model_score()} ')
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
