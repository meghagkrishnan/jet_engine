import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [
            'time_(cycles)', 'operational_setting_1', 'operational_setting_2',
            'T24_Total_temperature_at_LPC_outlet_(°R)', 'T30_Total_temperature_at_HPC_outlet_(°R)',
            'T50_Total_temperature_at_LPT_outlet_(°R)', 'P30_Total_pressure_at_HPC_outlet_(psia)',
            'Nf_Physical_fan_speed_(rpm)', 'Nc_Physical_core_speed_(rpm)', 'Ps30_Static_pressure_at_HPC_outlet_(psia)',
            'phi_Ratio_of_fuel_flow_to_Ps30_(pps/psi)', 'NRf_Corrected_fan_speed_(rpm)', 'NRc_Corrected_core_speed_(rpm)',
            'BPR_Bypass_Ratio', 'htBleed_Bleed_Enthalpy', 'W31_HPT_coolant_bleed_(lbm/s)', 'W32_LPT_coolant_bleed_(lbm/s)'
        ]),
        ('cat', OneHotEncoder(), ['unit_number'])
    ]
)
# Define the full pipeline including preprocessing and the model
pipeline_gbm = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor())
])

def train_gbm_with_pipeline(df: pd.DataFrame):
    """
    This function trains a Gradient Boosting Machine (GBM) model using a pipeline on the preprocessed DataFrame.

    Parameters:
    df (pd.DataFrame): Cleaned DataFrame with features and RUL column.

    Returns:
    Pipeline: Trained pipeline with preprocessing and GBM model.
    """

    # Separate features and target
    X = df.drop(['RUL'], axis=1)
    y = df['RUL']

    # Fit the pipeline
    pipeline_gbm.fit(X, y)

    # Make predictions and evaluate the model
    y_pred = pipeline.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error: {mse}")

    return pipeline_gbm



 #df = pd.DataFrame(data)

    # Train the model with the pipeline
   # trained_pipeline = train_gbm_with_pipeline(df)

#create a pipeline for training and one for testing, because not fit with testing

'''
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function normalizes the specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame to be normalized.

    Returns:
    pd.DataFrame: Normalized DataFrame.
    """
    # Normalize the data (excluding 'unit_number' and 'RUL')
    columns_to_normalize = df.columns.difference(['unit_number', 'RUL'])
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df
    '''
