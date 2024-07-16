#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_path = 'raw_data'
train_df = pd.read_csv(os.path.join(data_path, 'train1_filterd.csv'), sep=',')
test_df = pd.read_csv(os.path.join(data_path, 'test1_filterd.csv'), sep=',')


# In[4]:


print("Train DataFrame Info:")
print(train_df.info())
print("\nTest DataFrame Info:")
print(test_df.info())


# In[5]:


print("\nTrain DataFrame Head:")
print(train_df.head())
print("\nTest DataFrame Head:")
print(test_df.head())


# In[6]:


print("\nMissing values in Train DataFrame:")
print(train_df.isnull().sum())
print("\nMissing values in Test DataFrame:")
print(test_df.isnull().sum())


# In[7]:


print("\nTrain DataFrame Description:")
print(train_df.describe())
print("\nTest DataFrame Description:")
print(test_df.describe())


# In[8]:


for column in train_df.columns:
    train_df[column] = pd.to_numeric(train_df[column], errors='coerce')

print("\nTrain DataFrame Data Types After Conversion:")
print(train_df.dtypes)

print("\nMissing values in Train DataFrame After Conversion:")
print(train_df.isnull().sum())

train_df.dropna(inplace=True)

print("\nTrain DataFrame Info After Dropping NA:")
print(train_df.info())


# In[9]:


plt.figure(figsize=(20, 16))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[11]:


sensors_to_plot = [
    'T24_Total_temperature_at_LPC_outlet_(°R)',
    'T30_Total_temperature_at_HPC_outlet_(°R)',
    'T50_Total_temperature_at_LPT_outlet_(°R)',
    'P30_Total_pressure_at_HPC_outlet_(psia)',
    'Nf_Physical_fan_speed_(rpm)',
    'Nc_Physical_core_speed_(rpm)'
]

train_df[sensors_to_plot].hist(figsize=(15, 10))
plt.tight_layout()
plt.show()


# In[14]:


features = [
    'T24_Total_temperature_at_LPC_outlet_(°R)',
    'T30_Total_temperature_at_HPC_outlet_(°R)',
    'T50_Total_temperature_at_LPT_outlet_(°R)',
    'P30_Total_pressure_at_HPC_outlet_(psia)',
    'Nf_Physical_fan_speed_(rpm)',
    'Nc_Physical_core_speed_(rpm)'
]
target = 'RUL'

print(train_df[features + [target]].isnull().sum())


# In[18]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

scaler = StandardScaler()
X = scaler.fit_transform(train_df[features])
y = train_df[target]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)


# In[19]:


print(f"Training MSE: {train_mse}, Training R2: {train_r2}")
print(f"Validation MSE: {val_mse}, Validation R2: {val_r2}")


# In[20]:


plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.show()


# In[ ]:




