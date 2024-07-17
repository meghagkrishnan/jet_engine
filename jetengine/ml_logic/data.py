from jetengine.params import *


def read_data(path):
    df = pd.read_csv(path,sep = ' ', header=None)
    columns = COLUMN_NAMES
    df.columns = columns
    df = train_FD001.drop(columns = ['sm22', 'sm23'])

    return df

def clean_data_bm(df: pd.DataFrame):
