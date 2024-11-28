import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load the model and preprocessing objects
model = joblib.load('stacking_model.pkl')
selector = joblib.load('selector.pkl')  # Pre-saved SelectKBest
poly = joblib.load('poly.pkl')

st.title('Car Price Prediction App')

# Collect input data from the user
km_driven = st.number_input('Kilometers Driven', min_value=0)
mileage = st.number_input('Mileage (km/ltr/kg)', min_value=0.0)
engine = st.number_input('Engine (cc)', min_value=0)
max_power = st.number_input('Max Power (bhp)', min_value=0.0)
seats = st.number_input('Seats', min_value=0)
year = st.number_input('Year of Manufacture', min_value=1900, max_value=2024)

# Create a DataFrame from the input data
input_data = pd.DataFrame([{
    'km_driven': km_driven,
    'mileage(km/ltr/kg)': mileage,
    'engine': engine,
    'max_power': max_power,
    'seats': seats,
    'year': year
}])

# Feature Engineering
input_data['age'] = 2024 - input_data['year']
input_data['mileage_per_engine'] = input_data['mileage(km/ltr/kg)'] / input_data['engine']
input_data['log_km_driven'] = np.log1p(input_data['km_driven'])  # log1p to handle zero values
input_data['engine_squared'] = input_data['engine'] ** 2
input_data['max_power_cubed'] = input_data['max_power'] ** 3

# Drop the 'year' column as it's been transformed to 'age'
input_data.drop(['year'], axis=1, inplace=True)

# Handle NaNs in derived columns
input_data['mileage_per_engine'] = input_data['mileage_per_engine'].fillna(0)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
input_data_imputed = imputer.fit_transform(input_data)

# Convert back to DataFrame and set column names
input_data = pd.DataFrame(input_data_imputed, columns=input_data.columns)

# Apply SelectKBest selector (pre-saved)
input_data_selected = selector.transform(input_data)

# Apply polynomial features
input_data_poly = poly.transform(input_data_selected)

# Make prediction
if st.button('Predict'):
    try:
        prediction = model.predict(input_data_poly)
        st.write(f'Predicted Selling Price: {prediction[0]:,.2f} units')
    except ValueError as e:
        st.error(f"An error occurred during prediction: {e}")

# Optional: Display input data for verification
st.subheader('Input Data')
selected_columns = input_data.columns[selector.get_support()]
st.write(pd.DataFrame(input_data_selected, columns=selected_columns))
