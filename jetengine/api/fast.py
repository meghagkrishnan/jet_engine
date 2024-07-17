from fastapi import FastAPI
import pandas as pd
import numpy as np
from jetengine.ml_logic.preprocessor import process_Xdata
from jetengine.ml_logic.registry import load_LG_model

app = FastAPI()

app.state.model = load_LG_model()

@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }
#Request URL
#http://127.0.0.1:8000/predict?id=1&cycle=20&setting1=-0.0037&setting2=0.0001&setting3=100.0&T24_Total_temperature_at_LPC_outlet=643.04&T30_Total_temperature_at_HPC_outlet=1581.11&T50_Total_temperature_at_LPT_outlet=1402.23&P30_Total_pressure_at_HPC_outlet=554.81&Nf_Physical_fan_speed=2388.05&Nc_Physical_core_speed=9045.9&Ps30_Static_pressure_at_HPC_outlet=47.22&phi_Ratio_of_fuel_flow_to_Ps30=522.07&NRf_Corrected_fan_speed=2388.02&NRc_Corrected_core_speed=8129.71&BPR_Bypass_Ratio=8.421&htBleed_Bleed_Enthalpy=392&W31_HPT_coolant_bleed=39.03&W32_LPT_coolant_bleed=23.422
@app.get("/predict")
def predict(
        id: int = 1,
        cycle: int = 20,
        setting1: float = -0.0037,
        setting2: float = 0.0001,
        setting3: float =  100.0,
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
