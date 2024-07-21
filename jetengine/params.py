import os

LOCAL_DATA_PATH = "/home/meghagkrishnan/code/meghagkrishnan/jet_engine/raw_data"
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs", "jetengine")
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
LOCAL_REGISTRY_PATH = os.path.join(ROOT_DIR, 'models')
