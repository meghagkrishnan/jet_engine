import numpy as np
import pandas as pd

from pathlib import Path

from jetengine.params import *
from jetengine.ml_logic.data import clean_Xdata
from jetengine.ml_logic.model import model_evaluate, train_base_model
from jetengine.ml_logic.preprocessor import seperate_data,process_Xdata, split_data

data_query_cache_path = Path(LOCAL_DATA_PATH)
train_path = data_query_cache_path.joinpath("train1_filtered.csv")
#test_path = data_query_cache_path.joinpath("test1_filterd.csv")

def baseline_model_score():
    if train_path.is_file():
        print("Loading train data from local CSV...")

        train_data = pd.read_csv(train_path)
        #test_data = pd.read_csv(test_path)
    else:
        print("No data in the folder...")
    X,y = seperate_data(train_data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_cleaned = clean_Xdata(X_train)
    X_test_cleaned = clean_Xdata(X_test)
    X_train_processed = process_Xdata(X_train_cleaned)
    X_test_processed = process_Xdata(X_test_cleaned)

    baseline_model = train_base_model(X_train_processed, y_train)

    y_pred = baseline_model.predict(X_test_processed)
    scores = model_evaluate(y_test, y_pred)

    return scores

if __name__ == '__main__':
    try:
        print(f'Baseline score: {baseline_model_score()}')
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
