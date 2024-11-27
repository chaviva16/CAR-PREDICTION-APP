import yaml
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load configuration from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access configuration settings
model_path = config["model_path"]
selector_path = config["selector_path"]
poly_path = config["poly_path"]
app_settings = config["app_settings"]

# Load the model and other objects using the paths from the config file
model = joblib.load(model_path)
selector = joblib.load(selector_path)
poly = joblib.load(poly_path)

# Streamlit app title
st.title("Car Price Prediction App")

# Example input fields for user to provide data
km_driven = st.number_input("Kilometers Driven", min_value=0)
mileage = st.number_input("Mileage (km/l)", min_value=0.0)
engine = st.number_input("Engine Capacity (cc)", min_value=0)
max_power = st.number_input("Max Power (bhp)", min_value=0.0)
seats = st.number_input("Number of Seats", min_value=1)
year = st.number_input("Year of Manufacture", min_value=1900, max_value=2024, step=1)

# Collect input data
input_data = pd.DataFrame({
    'km_driven': [km_driven],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats],
    'year': [year]
})

# Preprocess input data
input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
input_data.dropna(inplace=True)
input_data = input_data.astype(float)

# Display input data
st.write("Input data for prediction:")
st.write(input_data)

# Make a prediction
if st.button("Predict"):
    try:
        transformed_input_data = poly.transform(selector.transform(input_data))
        prediction = model.predict(transformed_input_data)
        st.success(f"The predicted price of the car is: {prediction[0]:,.2f} units")
    except ValueError as e:
        st.error(f"An error occurred during prediction: {e}")
