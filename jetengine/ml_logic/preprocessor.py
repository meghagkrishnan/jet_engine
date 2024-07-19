import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def seperate_data(df):
    X = df.drop(columns = ['RUL'])
    y = df['RUL']
    return X, y

def process_Xdata(X):
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(features_normalized, columns=X.columns)

    return X_scaled

def split_data(X, y):

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X[X['id'] <= 80]
    X_test = X[X['id'] > 80]
    y_train = y[:len(X_train)]
    y_test = y[len(X_train):]


    return X_train, X_test, y_train, y_test
