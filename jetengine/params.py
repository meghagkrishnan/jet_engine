import os

LOCAL_DATA_PATH = "~/code/meghagkrishnan/jet_engine/jetengine/data"
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs", "jetengine")
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
LOCAL_REGISTRY_PATH = os.path.join(ROOT_DIR, 'models')
