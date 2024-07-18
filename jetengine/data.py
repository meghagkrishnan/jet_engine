
import pandas as pd

from jetengine import *


def clean_Xdata(df: pd.DataFrame):

    #Remove id,cycle and setting 3 column for training
    df_new = df.drop(columns = ['id','cycle','setting3'])

    return df_new


def clean_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function assigns column names to the DataFrame, drops specified columns,
    and adds a Remaining Useful Life (RUL) column.

    Parameters:
    df (pd.DataFrame): Input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: Cleaned DataFrame with RUL column added.
    """
    # Rename the columns
    columns = [
    'id',
    'cycle',
    'setting1',
    'setting2',
    'setting3',
    'T2_Total_temperature_at_fan_inlet',
    'T24_Total_temperature_at_LPC_outlet',
    'T30_Total_temperature_at_HPC_outlet',
    'T50_Total_temperature_at_LPT_outlet',
    'P2_Pressure_at_fan_inlet',
    'P15_Total_pressure_in_bypass_duct',
    'P30_Total_pressure_at_HPC_outlet',
    'Nf_Physical_fan_speed',
    'Nc_Physical_core_speed',
    'epr_Engine_pressure_ratio',
    'Ps30_Static_pressure_at_HPC_outlet',
    'phi_Ratio_of_fuel_flow_to_Ps30',
    'NRf_Corrected_fan_speed',
    'NRc_Corrected_core_speed',
    'BPR_Bypass_Ratio',
    'farB_Burner_fuel_air_ratio',
    'htBleed_Bleed_Enthalpy',
    'Nf_dmd_Demanded_fan_speed',
    'PCNfR_dmd_Demanded_corrected_fan_speed',
    'W31_HPT_coolant_bleed',
    'W32_LPT_coolant_bleed',
    'sm22',
    'sm23'
]

    # Assign the column names to the DataFrame
    df.columns = columns

    # Add Remaining Useful Life (RUL) column
    max_cycle = df.groupby('id')['cycle'].max()
    df = df.merge(max_cycle, on='id', suffixes=('', '_max'))
    df['RUL'] = df['cycle_max'] - df['cycle']
    df = df.drop(columns=['cycle_max'])

    # Columns to drop
    columns_to_drop = ['sm22', 'sm23', 'setting3', 'T2_Total_temperature_at_fan_inlet', 'P2_Pressure_at_fan_inlet', "P15_Total_pressure_in_bypass_duct",
            'epr_Engine_pressure_ratio', 'farB_Burner_fuel_air_ratio', 'Nf_dmd_Demanded_fan_speed',
            'PCNfR_dmd_Demanded_corrected_fan_speed']

    # Drop the specified columns
    cleaned_train_df= df.drop(columns_to_drop, axis=1)

    return cleaned_train_df


def clean_test_data(df: pd.DataFrame) -> pd.DataFrame:
     """
    This function assigns column names to the DataFrame, drops specified columns,
    and calculates and adds a Remaining Useful Life (RUL) column.

    Parameters:
    test_df (pd.DataFrame): Input test DataFrame to be cleaned.
    rul_df (pd.DataFrame): DataFrame containing RUL values for the test units.

    Returns:
    pd.DataFrame: Cleaned DataFrame with RUL column added.
    """
    # Rename the columns
    columns = [
    'id',
    'cycle',
    'setting1',
    'setting2',
    'setting3',
    'T2_Total_temperature_at_fan_inlet',
    'T24_Total_temperature_at_LPC_outlet',
    'T30_Total_temperature_at_HPC_outlet',
    'T50_Total_temperature_at_LPT_outlet',
    'P2_Pressure_at_fan_inlet',
    'P15_Total_pressure_in_bypass_duct',
    'P30_Total_pressure_at_HPC_outlet',
    'Nf_Physical_fan_speed',
    'Nc_Physical_core_speed',
    'epr_Engine_pressure_ratio',
    'Ps30_Static_pressure_at_HPC_outlet',
    'phi_Ratio_of_fuel_flow_to_Ps30',
    'NRf_Corrected_fan_speed',
    'NRc_Corrected_core_speed',
    'BPR_Bypass_Ratio',
    'farB_Burner_fuel_air_ratio',
    'htBleed_Bleed_Enthalpy',
    'Nf_dmd_Demanded_fan_speed',
    'PCNfR_dmd_Demanded_corrected_fan_speed',
    'W31_HPT_coolant_bleed',
    'W32_LPT_coolant_bleed',
    'sm22',
    'sm23']

    # Assign the column names to the DataFrame
    df.columns = columns

    # Get the max cycle for each unit in the test dataset
    max_cycle_test = df.groupby('id')['cycle'].max().reset_index()
    max_cycle_test.columns = ['id', 'max_cycle']

    # Merge with the RUL values
    max_cycle_test = max_cycle_test.merge(rul_df, left_index=True, right_index=True)

    # Calculate the RUL for each row in the test dataset
    test_df = df.merge(max_cycle_test[['id', 'max_cycle', 'RUL']], on='id')
    test_df['RUL'] = test_df['RUL'] + (test_df['max_cycle'] - test_df['cycle'])
    test_FD001 = test_df.drop(columns=['max_cycle'])

    # Columns to drop
    columns_to_drop = ['sm22', 'sm23', 'setting3', 'T2_Total_temperature_at_fan_inlet', 'P2_Pressure_at_fan_inlet', "P15_Total_pressure_in_bypass_duct",
            'epr_Engine_pressure_ratio', 'farB_Burner_fuel_air_ratio', 'Nf_dmd_Demanded_fan_speed',
            'PCNfR_dmd_Demanded_corrected_fan_speed']

    # Drop the specified columns
    cleaned_test_df = df.drop(columns_to_drop, axis=1)

    return cleaned_test_df



# Example usage:
# test_FD001 = pd.read_csv('path_to_test_FD001.csv', delim_whitespace=True, header=None)
# rul_FD001 = pd.read_csv('path_to_RUL_FD001.csv', delim_whitespace=True, header=None)

# cleaned_test_df = clean_test_data(test_FD001, rul_FD001)
# print(cleaned_test_df.head())
