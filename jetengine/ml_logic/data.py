from jetengine.params import *
import pandas as pd

def clean_Xdata(df: pd.DataFrame):

    #Remove id,cycle and setting 3 column for training
    df_new = df.drop(columns = ['id','cycle','setting3'])

    return df_new
