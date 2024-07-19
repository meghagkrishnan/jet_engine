from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from jetengine.ml_logic.registry import save_LG_model

def train_base_model(X_train, y_train):
    #Create a base model
    base_model = LinearRegression()

    # Train the model using the training sets
    base_model.fit(X_train, y_train)

    save_LG_model(base_model)

    return base_model

def model_evaluate(y_test, y_pred):

    R2_score = r2_score(y_test, y_pred)
    #print(f'R² Score: {r2_base}')

    mse = mean_squared_error(y_test, y_pred)
    print("✅ baseline model evaluation done")
    return R2_score, mse
