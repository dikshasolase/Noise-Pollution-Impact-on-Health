import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("noise_health_model.pkl")

# App title
st.title("🔊 Noise Pollution Health Risk Predictor")

st.write("Predict health risk due to long-term noise exposure")

# User inputs
noise_level_db = st.slider("Noise Level (dB)", 30, 120, 70)
traffic_density = st.slider("Traffic Density", 0, 100, 50)

area_type = st.selectbox(
    "Area Type",
    ["Residential", "Commercial", "Industrial"]
)

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Night"]
)

# Encode categorical inputs (same logic as training)
area_map = {"Residential": 0, "Commercial": 1, "Industrial": 2}
time_map = {"Morning": 0, "Afternoon": 1, "Night": 2}

area_encoded = area_map[area_type]
time_encoded = time_map[time_of_day]

# Prediction button
if st.button("Predict Health Risk"):
    input_data = np.array([[noise_level_db,
                            traffic_density,
                            area_encoded,
                            time_encoded]])

    prediction = model.predict(input_data)

    st.success(f"🔮 Predicted Health Risk Score: {prediction[0]:.2f}")
