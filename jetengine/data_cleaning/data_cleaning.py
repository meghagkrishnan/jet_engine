import pandas as pd
from sklearn.preprocessing import StandardScaler


#train_FD001 = pd.read_csv('train_FD001.txt',sep=" ",header=None)

#test_FD001 = pd.read_csv("test_FD001.txt",sep=" ",header=None)

#RUL_FD001 = pd.read_csv("RUL_FD001.txt", header=None)

def cleaning_training_data(df: pd.DataFrame):
    """
    This function assigns specific column names to the DataFrame, drops specified columns,
    and adds a Remaining Useful Life (RUL) column.

    Parameters:
    df (pd.DataFrame): Input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: Cleaned DataFrame with RUL column added.
    """
    columns = [
        'unit_number', 'time_(cycles)', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
        'T2_Total_temperature_at_fan_inlet_(°R)', 'T24_Total_temperature_at_LPC_outlet_(°R)',
        'T30_Total_temperature_at_HPC_outlet_(°R)', 'T50_Total_temperature_at_LPT_outlet_(°R)',
        'P2_Pressure_at_fan_inlet_(psia)', 'P15_Total_pressure_in_bypass-duct_(psia)',
        'P30_Total_pressure_at_HPC_outlet_(psia)', 'Nf_Physical_fan_speed_(rpm)', 'Nc_Physical_core_speed_(rpm)',
        'epr_Engine_pressure_ratio_(P50/P2)', 'Ps30_Static_pressure_at_HPC_outlet_(psia)',
        'phi_Ratio_of_fuel_flow_to_Ps30_(pps/psi)', 'NRf_Corrected_fan_speed_(rpm)',
        'NRc_Corrected_core_speed_(rpm)', 'BPR_Bypass_Ratio', 'farB_Burner_fuel-air_ratio', 'htBleed_Bleed_Enthalpy',
        'Nf_dmd_Demanded_fan_speed_(rpm)', 'PCNfR_dmd_Demanded_corrected_fan_speed_(rpm)',
        'W31_HPT_coolant_bleed_(lbm/s)', 'W32_LPT_coolant_bleed_(lbm/s)', 'Sensor_26', 'Sensor_27'
    ]

    # Assign the column names to the DataFrame
    df.columns = columns

    # Columns to drop
    columns_to_drop = ['Sensor_26', 'Sensor_27']

    # Additional columns to drop
    no_unique_column = [
        'operational_setting_3', 'T2_Total_temperature_at_fan_inlet_(°R)', 'P2_Pressure_at_fan_inlet_(psia)',
        'P15_Total_pressure_in_bypass-duct_(psia)', 'epr_Engine_pressure_ratio_(P50/P2)', 'farB_Burner_fuel-air_ratio',
        'Nf_dmd_Demanded_fan_speed_(rpm)', 'PCNfR_dmd_Demanded_corrected_fan_speed_(rpm)'
    ]

    # Combine all columns to drop
    all_columns_to_drop = columns_to_drop + no_unique_column

    # Drop the specified columns
    df = df.drop(all_columns_to_drop, axis=1)

    # Add Remaining Useful Life (RUL) column
    df_train = df['RUL'] = df.groupby('unit_number')['time_(cycles)'].apply(lambda x: x.max() - x).values

    return df_train

def cleaning_testing_data(df: pd.DataFrame):
    """
    This function assigns specific column names to the DataFrame, drops specified columns,
    and adds a Remaining Useful Life (RUL) column.

    Parameters:
    df (pd.DataFrame): Input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: Cleaned DataFrame without RUL column added.
    """
    columns = [
        'unit_number', 'time_(cycles)', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
        'T2_Total_temperature_at_fan_inlet_(°R)', 'T24_Total_temperature_at_LPC_outlet_(°R)',
        'T30_Total_temperature_at_HPC_outlet_(°R)', 'T50_Total_temperature_at_LPT_outlet_(°R)',
        'P2_Pressure_at_fan_inlet_(psia)', 'P15_Total_pressure_in_bypass-duct_(psia)',
        'P30_Total_pressure_at_HPC_outlet_(psia)', 'Nf_Physical_fan_speed_(rpm)', 'Nc_Physical_core_speed_(rpm)',
        'epr_Engine_pressure_ratio_(P50/P2)', 'Ps30_Static_pressure_at_HPC_outlet_(psia)',
        'phi_Ratio_of_fuel_flow_to_Ps30_(pps/psi)', 'NRf_Corrected_fan_speed_(rpm)',
        'NRc_Corrected_core_speed_(rpm)', 'BPR_Bypass_Ratio', 'farB_Burner_fuel-air_ratio', 'htBleed_Bleed_Enthalpy',
        'Nf_dmd_Demanded_fan_speed_(rpm)', 'PCNfR_dmd_Demanded_corrected_fan_speed_(rpm)',
        'W31_HPT_coolant_bleed_(lbm/s)', 'W32_LPT_coolant_bleed_(lbm/s)', 'Sensor_26', 'Sensor_27'
    ]

    # Assign the column names to the DataFrame
    df.columns = columns

    # Columns to drop (26th and 27th in zero-indexing)
    columns_to_drop = ['Sensor_26', 'Sensor_27']

    # Additional columns to drop
    no_unique_column = [
        'operational_setting_3', 'T2_Total_temperature_at_fan_inlet_(°R)', 'P2_Pressure_at_fan_inlet_(psia)',
        'P15_Total_pressure_in_bypass-duct_(psia)', 'epr_Engine_pressure_ratio_(P50/P2)', 'farB_Burner_fuel-air_ratio',
        'Nf_dmd_Demanded_fan_speed_(rpm)', 'PCNfR_dmd_Demanded_corrected_fan_speed_(rpm)'
    ]

    # Combine all columns to drop
    all_columns_to_drop = columns_to_drop + no_unique_column

    # Drop the specified columns
    df_test = df.drop(all_columns_to_drop, axis=1)

    return df_test


'''
Testing
    # Clean
    cleaned_df = cleaning_data(df)

    # Preprocess (normalize)
    preprocessed_df = preprocess_data(cleaned_df)

    print(preprocessed_df)
    '''
