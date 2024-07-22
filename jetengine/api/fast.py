from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import pandas as pd
import numpy as np

from jetengine.ml_logic.preprocessor import process_Xdata
from jetengine.ml_logic.registry import load_LG_model
from jetengine.ml_logic.data import display_test_data, clean_Xdata

app = FastAPI()

app.state.model = load_LG_model()

@app.get("/")
def root():
    return {
        'message': "Hi, Welcome to Jet Engine API!"
    }


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, sep = ' ', header=None)
    df_clean = display_test_data(df)
    file.file.close()
    #first_five_rows = df_clean.head().to_dict(orient='records')
    first_row = df_clean.iloc[0].to_dict()

    df_predict = clean_Xdata(df_clean)

    model = app.state.model
    assert model is not None


    X_processed = process_Xdata(df_predict)
    y_pred = model.predict(X_processed)

    return JSONResponse(content={
            "Data": first_row,
            "RUL": np.round(float(y_pred))
        })
