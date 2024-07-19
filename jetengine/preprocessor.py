import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Separate features and target
X = df.drop(['RUL'], axis=1)
y = df['RUL']


# Create a preprocessor
preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a baseline model
baseline_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


# Preprocessing pipeline for test data ????
pipeline_train_base = Pipeline([
    ('scaler', StandardScaler()),

])


 #Make a prediction with pipeline
def train_model_with_pipe(df: pd.DataFrame):
    """
    This function trains a linear regression model using a pipeline on the cleaned DataFrame.

    Parameters:
    df (pd.DataFrame): Cleaned DataFrame with features and RUL column.

    Returns:
    Pipeline: Trained pipeline with preprocessing and linear regression model.
    """

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline
    pipeline_train_base.fit(X_train, y_train)



    return pipeline_train_base

y_pred_base = pipeline_train_base.predict(X_test)
# Make predictions and evaluate the model

    r2_base = r2_score(y_test, y_pred_base)
    print(f'R² Score: {r2_base}')
cross_val_score(pipeline_train_base, X, y, cv=10, scoring='r2').mean()

 #df = pd.DataFrame(data)

    # Train the model with the pipeline
   # trained_pipeline = train_gbm_with_pipeline(df)

#create a pipeline for training and one for testing, because not fit with testing
