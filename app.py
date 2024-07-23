import streamlit as st
from PIL import Image
import pandas as pd
import requests

# Title of the app
st.title('Prediction of RUL for Jet Engines')

# Display the heading image
heading_image = Image.open("images/image.png")

st.image(heading_image, use_column_width=True)

uploaded_file = st.file_uploader("Choose a text file", type="txt")

jetengine_api_url = 'http://127.0.0.1:8000/predict'
column_names = [
                'id',
                'cycle',
                'setting1',
                'setting2',
                'setting3',
                'T2_Total_temperature_at_fan_inlet',
                'T24_Total_temperature_at_LPC_outlet',
                'T30_Total_temperature_at_HPC_outlet',
                'T50_Total_temperature_at_LPT_outlet',
                'P2_Pressure_at_fan_inlet',
                'P15_Total_pressure_in_bypass_duct',
                'P30_Total_pressure_at_HPC_outlet',
                'Nf_Physical_fan_speed',
                'Nc_Physical_core_speed',
                'epr_Engine_pressure_ratio',
                'Ps30_Static_pressure_at_HPC_outlet',
                'phi_Ratio_of_fuel_flow_to_Ps30',
                'NRf_Corrected_fan_speed',
                'NRc_Corrected_core_speed',
                'BPR_Bypass_Ratio',
                'farB_Burner_fuel_air_ratio',
                'htBleed_Bleed_Enthalpy',
                'Nf_dmd_Demanded_fan_speed',
                'PCNfR_dmd_Demanded_corrected_fan_speed',
                'W31_HPT_coolant_bleed',
                'W32_LPT_coolant_bleed',
                ]

if uploaded_file is not None:
    # Read the data from the text file
    data = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
    data.columns = column_names
    data = data.drop(columns = ['id','cycle','setting3','T2_Total_temperature_at_fan_inlet','P2_Pressure_at_fan_inlet',
                                'P15_Total_pressure_in_bypass_duct','epr_Engine_pressure_ratio','farB_Burner_fuel_air_ratio',
                                'Nf_dmd_Demanded_fan_speed','PCNfR_dmd_Demanded_corrected_fan_speed'])
    data_to_predict = data[0:1]
    setting1 = data_to_predict['setting1']
    setting2 = data_to_predict['setting2'],
    T24_Total_temperature_at_LPC_outlet = data_to_predict['T24_Total_temperature_at_LPC_outlet'],
    T30_Total_temperature_at_HPC_outlet = data_to_predict['T30_Total_temperature_at_HPC_outlet'],
    T50_Total_temperature_at_LPT_outlet = data_to_predict['T50_Total_temperature_at_LPT_outlet'],
    P30_Total_pressure_at_HPC_outlet = data_to_predict['P30_Total_pressure_at_HPC_outlet'],
    Nf_Physical_fan_speed = data_to_predict['Nf_Physical_fan_speed'],
    Nc_Physical_core_speed = data_to_predict['Nc_Physical_core_speed'],
    Ps30_Static_pressure_at_HPC_outlet = data_to_predict['Ps30_Static_pressure_at_HPC_outlet'],
    phi_Ratio_of_fuel_flow_to_Ps30 = data_to_predict['phi_Ratio_of_fuel_flow_to_Ps30'],
    NRf_Corrected_fan_speed = data_to_predict['NRf_Corrected_fan_speed'],
    NRc_Corrected_core_speed = data_to_predict['NRc_Corrected_core_speed'],
    BPR_Bypass_Ratio = data_to_predict['BPR_Bypass_Ratio'],
    htBleed_Bleed_Enthalpy = data_to_predict['htBleed_Bleed_Enthalpy'],
    W31_HPT_coolant_bleed = data_to_predict['W31_HPT_coolant_bleed'],
    W32_LPT_coolant_bleed = data_to_predict['W32_LPT_coolant_bleed']

    params = dict(
        setting1= setting1,
        setting2= setting2,
        T24_Total_temperature_at_LPC_outlet= T24_Total_temperature_at_LPC_outlet,
        T30_Total_temperature_at_HPC_outlet= T30_Total_temperature_at_HPC_outlet,
        T50_Total_temperature_at_LPT_outlet= T50_Total_temperature_at_LPT_outlet,
        P30_Total_pressure_at_HPC_outlet= P30_Total_pressure_at_HPC_outlet,
        Nf_Physical_fan_speed= Nf_Physical_fan_speed,
        Nc_Physical_core_speed= Nc_Physical_core_speed,
        Ps30_Static_pressure_at_HPC_outlet= Ps30_Static_pressure_at_HPC_outlet,
        phi_Ratio_of_fuel_flow_to_Ps30= phi_Ratio_of_fuel_flow_to_Ps30,
        NRf_Corrected_fan_speed= NRf_Corrected_fan_speed,
        NRc_Corrected_core_speed= NRc_Corrected_core_speed,
        BPR_Bypass_Ratio= BPR_Bypass_Ratio,
        htBleed_Bleed_Enthalpy= htBleed_Bleed_Enthalpy,
        W31_HPT_coolant_bleed= W31_HPT_coolant_bleed,
        W32_LPT_coolant_bleed= W32_LPT_coolant_bleed
        )

    response = requests.get(jetengine_api_url, params=params)

    if response.status_code == 200:
        prediction = response.json()
        pred = prediction['RUL']
        st.header(f'The given jet engine has {pred} more cycles before failure')
    else:
        st.header('Prediction failed. Please try again.')
else:
    st.header('Please upload a text file with engine data.')
