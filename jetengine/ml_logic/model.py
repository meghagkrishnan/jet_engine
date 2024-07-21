from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from jetengine.ml_logic.registry import save_LG_model
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, Normalization, LSTM

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def train_base_model(X_train, y_train):
    #Create a base model
    base_model = LinearRegression()

    # Train the model using the training sets
    base_model.fit(X_train, y_train)

    save_LG_model(base_model)
    #print("✅ Saved Baseline model")
    return base_model

def LR_model_evaluate(y_test, y_pred):

    R2_score = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("✅ Baseline model evaluation done")

    return R2_score, rmse
