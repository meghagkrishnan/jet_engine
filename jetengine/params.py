import os

import numpy as np
#train_FD001 = pd.read_csv('train_FD001.txt',sep=" ",header=None)

#test_FD001 = pd.read_csv("test_FD001.txt",sep=" ",header=None)

#RUL_FD001 = pd.read_csv("RUL_FD001.txt", header=None)


##################  VARIABLES  ##################
DATA_SIZE = "1k" # ["1k", "200k", "all"]
CHUNK_SIZE = 200
GCP_PROJECT = "<your project id>" # TO COMPLETE
GCP_PROJECT_WAGON = "wagon-public-datasets"
BQ_DATASET = "taxifare"
BQ_REGION = "EU"
MODEL_TARGET = "local"
##################  CONSTANTS  #####################

COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

DTYPES_RAW = {
    "fare_amount": "float32",
    "pickup_datetime": "datetime64[ns, UTC]",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16"
}

DTYPES_PROCESSED = np.float32
LOCAL_DATA_PATH = "~/code/meghagkrishnan/jet_engine/jetengine/data"
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs", "jetengine")
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
LOCAL_REGISTRY_PATH = os.path.join(ROOT_DIR, 'models')
