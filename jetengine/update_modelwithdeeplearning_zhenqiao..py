import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

file_path = os.path.expanduser("~/code/meghagkrishnan/jet_engine/raw_data/train1_update.csv")

train_FD001 = pd.read_csv(file_path, sep=',')

columns = ['id', 'cycle', 'setting1', 'setting2', 'T24_Total_temperature_at_LPC_outlet',
           'T30_Total_temperature_at_HPC_outlet', 'T50_Total_temperature_at_LPT_outlet',
           'P30_Total_pressure_at_HPC_outlet', 'Nf_Physical_fan_speed',
           'Nc_Physical_core_speed', 'Ps30_Static_pressure_at_HPC_outlet',
           'phi_Ratio_of_fuel_flow_to_Ps30', 'NRf_Corrected_fan_speed',
           'NRc_Corrected_core_speed', 'BPR_Bypass_Ratio', 'htBleed_Bleed_Enthalpy',
           'W31_HPT_coolant_bleed', 'W32_LPT_coolant_bleed']

assert list(train_FD001.columns) == columns
print(train_FD001.head())

max_cycle = train_FD001['cycle'].max()
train_FD001['RUL'] = max_cycle - train_FD001['cycle']

print(train_FD001['RUL'].describe())



plt.figure(figsize=(20,16))
sns.heatmap(train_FD001.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


correlation_threshold = 0.5
selected_features = [col for col in train_FD001.columns if col not in ['id', 'RUL', 'cycle']]
selected_features = train_FD001.corr()['RUL'][abs(train_FD001.corr()['RUL']) > correlation_threshold].index.tolist()
selected_features.remove('RUL')
if 'cycle' in selected_features:
    selected_features.remove('cycle')
print("Selected features:", selected_features)

scaler = StandardScaler()
train_FD001_scaled = train_FD001.copy()
train_FD001_scaled[selected_features] = scaler.fit_transform(train_FD001[selected_features])
train_FD001_cleaned = train_FD001_scaled.dropna()
print(train_FD001_cleaned.head())
print(train_FD001_cleaned.columns)


test_file_path = os.path.expanduser("~/code/meghagkrishnan/jet_engine/raw_data/test_FD001_processed.csv")
test_FD001 = pd.read_csv(test_file_path)
print("Initial test data columns:", test_FD001.columns)
print("Initial test data shape:", test_FD001.shape)

test_FD001_scaled = test_FD001.copy()
test_FD001_scaled[selected_features] = scaler.transform(test_FD001[selected_features])
test_FD001_cleaned = test_FD001_scaled.dropna()
print("\nFinal test data shape:", test_FD001_cleaned.shape)
print("Final test data columns:", test_FD001_cleaned.columns)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


X = train_FD001_cleaned.drop(['RUL', 'cycle'], axis=1)
y = train_FD001_cleaned['RUL']

X_train = X[X['id'] <= 80]
X_test = X[X['id'] > 80]
y_train = y[:len(X_train)]
y_test = y[len(X_train):]

print (X_train, X_test, y_train, y_test)


X_train = X_train.drop('id', axis=1)
X_test = X_test.drop('id', axis=1)

numeric_features = X_train.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)

y_test_pred = pipeline.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print("Test R2 Score:", test_r2)

X_test = test_FD001_cleaned.drop(['cycle'], axis=1)
X_test = X_test[X_train.columns]

y_test_pred = pipeline.predict(X_test)

results_df = pd.DataFrame({
    'id': test_FD001_cleaned['id'],  
    'cycle': test_FD001_cleaned['cycle'],
    'predicted_RUL': y_test_pred
})

final_predictions = results_df.groupby('id').last().reset_index()

print("\nFinal predictions:")
print(final_predictions.head())


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

columns_to_drop = ['RUL', 'cycle']
if 'id' in train_FD001_cleaned.columns:
    columns_to_drop.append('id')
X_train = train_FD001_cleaned.drop(columns_to_drop, axis=1)
y_train = train_FD001_cleaned['RUL']

columns_to_drop = ['cycle']
if 'id' in test_FD001_cleaned.columns:
    columns_to_drop.append('id')
X_test = test_FD001_cleaned.drop(columns_to_drop, axis=1)

X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

param_dist = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

random_search = RandomizedSearchCV(rf_pipeline, param_distributions=param_dist, 
                                   n_iter=10, cv=3, scoring='neg_mean_squared_error', 
                                   n_jobs=-1, random_state=42)
random_search.fit(X_sample, y_sample)

print("Best parameters:", random_search.best_params_)


best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)

if 'id' in test_FD001_cleaned.columns:
    results_df = pd.DataFrame({
        'id': test_FD001_cleaned['id'],
        'predicted_RUL': y_pred
    })
else:
    results_df = pd.DataFrame({
        'predicted_RUL': y_pred
    })

if 'id' in results_df.columns:
    final_predictions = results_df.groupby('id').last().reset_index()
else:
    final_predictions = results_df

print("\nFinal predictions:")
print(final_predictions.head())

final_predictions.to_csv('random_forest_predictions.csv', index=False)
cv_mse_scores = -cross_val_score(best_model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
cv_r2_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='r2')

print("Cross-validation MSE scores:", cv_mse_scores)
print("Average CV MSE:", cv_mse_scores.mean())

print("\nCross-validation R² scores:", cv_r2_scores)
print("Average CV R²:", cv_r2_scores.mean())
