import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("model", "house_price_model.pkl")
data = joblib.load(MODEL_PATH)

model = data['model']
scaler = data['scaler']
features = data['features']

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction System")
st.write("Predict house prices based on selected features. **For educational purposes only.**")

st.sidebar.header("Input House Features")

OverallQual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.sidebar.number_input("Ground Living Area (sq ft)", min_value=200, max_value=10000, step=50)
TotalBsmtSF = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, step=10)
GarageCars = st.sidebar.slider("Number of Cars in Garage", 0, 5, 1)
FullBath = st.sidebar.slider("Number of Full Bathrooms", 0, 5, 1)
Neighborhood = st.sidebar.selectbox("Neighborhood", ["NAmes","CollgCr","OldTown","Edwards","Somerst"])

# Prepare input
input_dict = {
    "OverallQual": OverallQual,
    "GrLivArea": GrLivArea,
    "TotalBsmtSF": TotalBsmtSF,
    "GarageCars": GarageCars,
    "FullBath": FullBath,
    "Neighborhood": Neighborhood
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df, columns=['Neighborhood'], drop_first=True)

# Ensure all columns match training
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[features]

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")
