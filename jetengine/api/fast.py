from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import pandas as pd
import numpy as np

from jetengine.ml_logic.preprocessor import process_Xdata
from jetengine.ml_logic.registry import load_LG_model
from jetengine.ml_logic.data import display_test_data

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

    df_predict = df_clean.drop(columns = ['id', 'cycle'])

    model = app.state.model
    assert model is not None


    X_processed = process_Xdata(df_predict)
    y_pred = model.predict(X_processed)

    return JSONResponse(content={
            "Data": first_row,
            "RUL": np.round(float(y_pred))
        })


'''
@app.get("/predict")
def predict(
        setting1: float = -0.0037,
        setting2: float = 0.0001,
        T24_Total_temperature_at_LPC_outlet: float = 643.04,
        T30_Total_temperature_at_HPC_outlet: float = 1581.11,
        T50_Total_temperature_at_LPT_outlet: float =  1405.23,
        P30_Total_pressure_at_HPC_outlet: float = 554.81,
        Nf_Physical_fan_speed: float = 2388.05,
        Nc_Physical_core_speed: float = 9045.9,
        Ps30_Static_pressure_at_HPC_outlet: float = 47.22,
        phi_Ratio_of_fuel_flow_to_Ps30: float = 522.07,
        NRf_Corrected_fan_speed: float = 2388.02,
        NRc_Corrected_core_speed: float = 8129.71,
        BPR_Bypass_Ratio: float = 8.421,
        htBleed_Bleed_Enthalpy: float = 392,
        W31_HPT_coolant_bleed: float = 39.03,
        W32_LPT_coolant_bleed: float = 23.422,
        ):
    X_pred = pd.DataFrame(locals(), index=[0])

    model = app.state.model
    assert model is not None

    X_processed = process_Xdata(X_pred)
    y_pred = model.predict(X_processed)

    return dict(RUL=np.round(float(y_pred)))

'''
