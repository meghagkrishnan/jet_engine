import glob
import os
import time
import pickle
import joblib

from jetengine.params import *
from keras import models

#from colorama import Fore, Style
from typing import Tuple

def save_LR_model(model):
    model_path = os.path.join(LOCAL_REGISTRY_PATH, 'base_model.pkl')
    #Save model
    joblib.dump(model, model_path)
    print("\n✅ Baseline Model saved locally")

def load_LR_model():
    model_path = os.path.join(LOCAL_REGISTRY_PATH, 'base_model.pkl')
    #Load model
    model = joblib.load(model_path)
    print("\n✅ Loaded Baseline Model")

    return model


def save_LSTM_model(model):

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, f"LSTM_{timestamp}")
    model.save(model_path, save_format = 'tf')
    #models.save_model(model, model_path)
    print("\n✅ LSTM Model saved locally")


def load_LSTM_model():

    #print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
    print("\n✅ Loading LSTM model from local registry...")
    # Get the latest model version name by the timestamp on disk
    #local_model_directory = os.path.join(LOCAL_REGISTRY_PATH)
    local_model_paths = os.path.join(LOCAL_REGISTRY_PATH, "LSTM_20240723-160846")
    #local_model_paths = glob.glob(f"{LOCAL_REGISTRY_PATH}/*")

    if not local_model_paths:
        return None
    #most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    #print("\n✅ Load model from disk..." )

    latest_model = models.load_model(local_model_paths, compile = True)

    print("\n✅ Model loaded from local registry")

    return latest_model
