import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('energy_model.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Energy Predictor", page_icon="⚡", layout="wide")

st.markdown("<h1 style='text-align:center;'>⚡ Energy Consumption Predictor</h1>",unsafe_allow_html=True)
st.write("<p style='text-align:center;margin-bottom:40px;'>Enter the building details to predict hourly energy usage.</p>",unsafe_allow_html=True)
# st.title("⚡ Energy Consumption Predictor")
# st.write("Enter the building details to predict hourly energy usage.")

col1,col2 = st.columns(2)

with col1:
    temp = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    sq_ft = st.number_input("Square Footage", value=1500.0)
    occupancy = st.number_input("Occupancy (People)", value=5)

with col2:
    renewable = st.number_input("Renewable Energy (kWh)", value=0.0)
    hvac = st.selectbox("HVAC Status", ["Off", "On"])
    lights = st.selectbox("Lighting Status", ["Off", "On"])
    hour = st.slider("Hour of Day", 0, 23, 12)

# Logic to process inputs to match your X.columns
# (Must match the exact order and names you used for training)
hvac_on = 1 if hvac == "On" else 0
lights_on = 1 if lights == "On" else 0

# Handle Cyclical Encoding for Hour
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# Create the input dataframe
input_data = pd.DataFrame({
    'Temperature': [temp],
    'Humidity': [humidity],
    'SquareFootage': [sq_ft],
    'Occupancy': [occupancy],
    'RenewableEnergy': [renewable],
    'DayOfWeek': [0], # Defaulting to Monday for simplicity, or add a selector
    'Temp_per_Occupancy': [temp * occupancy],
    'HVACUsage_On': [hvac_on],
    'LightingUsage_On': [lights_on],
    'hour_sin': [hour_sin],
    'hour_cos': [hour_cos],
    'month_sin': [0], # Defaulting or add selector
    'month_cos': [1],
    'day_sin': [0],
    'day_cos': [1]
})

# 1. Scale the input
input_scaled = scaler.transform(input_data)

# 2. Predict
if st.button("Predict Consumption"):
    prediction = model.predict(input_scaled)

    final_pred = prediction[0]
    
    # st.success(f"Estimated Energy Consumption: {final_pred:.2f} kWh")

    if final_pred > 90:
        st.warning(f"⚠️ High energy demand detected : {final_pred:.2f} kWh")
    else:
        st.success(f"✅ Normal energy demand : {final_pred:.2f} kWh")