from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import pandas as pd
import numpy as np

from jetengine.ml_logic.preprocessor import process_Xdata
from jetengine.ml_logic.registry import load_LR_model, load_LSTM_model
from jetengine.ml_logic.data import display_test_data, clean_Xdata, get_last_cycles

app = FastAPI()

#app.state.model.LR = load_LR_model()
app.state.model = load_LSTM_model()

@app.get("/")
def root():
    return {
        'message': "Hi, Welcome to Jet Engine API!"
    }

@app.post("/predictLSTM")
def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, sep = ' ', header=None)
    df_clean = display_test_data(df)
    file.file.close()

    # Extract last 50 rows from the input data

    last_rows_df = get_last_cycles(df_clean)

    if last_rows_df.empty:
        return JSONResponse(content={
            "Engine ID": engine_number,
            "Message": "Not enough data for prediction"
        })

    else:
        last_row = last_rows_df.iloc[-1].to_dict()
        df_cleaned = clean_Xdata(last_rows_df)
        df_predict = np.expand_dims(df_cleaned, axis =0)

        model = app.state.model
        assert model is not None

        y_pred_LSTM = model.predict(df_predict)

        return JSONResponse(content={
            "Data": last_row,
            "Message" : "Prediction Success",
            "RUL": np.round(float(y_pred_LSTM))
        })
