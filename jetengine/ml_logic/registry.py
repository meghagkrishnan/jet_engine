import glob
import os
import time
import pickle
import joblib

from jetengine.params import *

def save_LG_model(model):
    model_path = os.path.join(LOCAL_REGISTRY_PATH, 'base_model.pkl')
    #Save model
    joblib.dump(model, model_path)
    print("✅ Baseline Model saved locally")

def load_LG_model():
    model_path = os.path.join(LOCAL_REGISTRY_PATH, 'base_model.pkl')
    #Load model
    model = joblib.load(model_path)
    print("✅ Loaded Baseline Model")

    return model


def save_model(model):

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")


def load_model():

    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return latest_model
