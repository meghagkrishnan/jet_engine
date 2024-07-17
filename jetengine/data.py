
import pandas as pd

from jetengine import *


def cleaning_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function assigns column names to the DataFrame, drops specified columns,
    and adds a Remaining Useful Life (RUL) column.

    Parameters:
    df (pd.DataFrame): Input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: Cleaned DataFrame with RUL column added.
    """
    # Rename the columns
    columns = ['unit_number','time_in_cycles','os1', 'os2','os3', 'sm1', 'sm2', 'sm3', 'sm4', 'sm5',
               'sm6', 'sm7', 'sm8', 'sm9', 'sm10', 'sm11', 'sm12','sm13', 'sm14','sm15', 'sm16', 'sm17',
               'sm18', 'sm19','sm20', 'sm21', 'sm22', 'sm23']

    # Assign the column names to the DataFrame
    df.columns = columns

    # Columns to drop
    columns_to_drop = ['sm22', 'sm23', 'os3', 'sm1', 'sm5', 'sm6','sm10', 'sm16', 'sm18', 'sm19']

    # Drop the specified columns
    df = df.drop(columns_to_drop, axis=1)

    # Add Remaining Useful Life (RUL) column
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max()
    df = df.merge(max_cycles, on='unit_number', suffixes=('', '_max'))
    df['RUL'] = df['time_in_cycles_max'] - df['time_in_cycles']
    df = df.drop(columns=['time_in_cycles_max'])

    return df
