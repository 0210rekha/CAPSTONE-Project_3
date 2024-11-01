import numpy as np
import joblib
import pandas as pd
import streamlit as st

# Load the trained model
try:
    model = joblib.load('best_used_car_price_model.joblib')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")

# Load the data to get OEM options
def load_data():
    return pd.read_csv('car_data.csv')

data = load_data()

# Extract unique OEMs from the data
oem_options = data['OEM'].unique()

# Define encoding mappings for categorical features
fuel_type_mapping = {
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2,
    "Electric": 3,
    "LPG": 4
}

body_type_mapping = {
    "Convertibles": 0,
    "Coupe": 1,
    "Hatchback": 2,
    "Hybrids": 3,
    "Minivans": 4,
    "MUV": 5,
    "Pickup Trucks": 6,
    "Sedan": 7,
    "SUV": 8,
    "Wagon": 9
}

transmission_mapping = {
    "Manual": 0,
    "Automatic": 1
}

insurance_validity_mapping = {
    "Valid": 0,
    "Expired": 1,
    "Not Available": 2
}

# Streamlit user interface
st.title("Car Dheko - Used Car Price Prediction")
st.write("Enter car details to get a real-time price prediction")

# Main Car Details Input
st.header("Main Car Details")

# Organizing input into two columns
col1, col2 = st.columns(2)

with col1:
    model_year = st.slider("Model Year", 1990, 2024)
    fuel_type = st.selectbox("Fuel Type", list(fuel_type_mapping.keys()))
    kms_driven = st.number_input("Kms Driven", min_value=0, max_value=500000, step=500)
    owner_number = st.slider("Owner Number (Previous Owners)", 0, 5)
    engine_displacement = st.number_input("Engine Displacement (cc)", min_value=500, max_value=6000, step=100)

with col2:
    body_type = st.selectbox("Body Type", list(body_type_mapping.keys()))
    transmission = st.selectbox("Transmission", list(transmission_mapping.keys()))
    max_power = st.number_input("Max Power (bhp)", min_value=50, max_value=1000, step=10)
    torque = st.text_input("Torque")

    # Ensure torque is numeric; handle empty input
    try:
        torque_value = float(torque) if torque else 0.0  # Default to 0.0 if empty
    except ValueError:
        st.error("Please enter a valid numeric value for Torque.")
        torque_value = 0.0  # Default to 0.0 or handle as needed

    city = st.selectbox("City", data['City'].unique())

# Select OEM and model
oem = st.selectbox("OEM (Car Manufacturer)", oem_options)
model_options = data[data['OEM'] == oem]['Model'].unique()
model_name = st.selectbox("Model", model_options)
variant_options = data[(data['OEM'] == oem) & (data['Model'] == model_name)]['Variant Name'].unique()
variant_name = st.selectbox("Variant Name", variant_options)

# Optional Features
if st.checkbox("Include Optional Features"):
    st.subheader("Optional Details")
    seating_capacity = st.number_input("Seating Capacity", min_value=2, max_value=10, step=1)
    insurance_validity = st.selectbox("Insurance Validity", list(insurance_validity_mapping.keys()))
    mileage = st.number_input("Mileage (kmpl)", min_value=5, max_value=50, step=1)
    age_of_car = 2024 - int(model_year)
else:
    seating_capacity = 0
    insurance_validity = "Valid"
    mileage = 0
    age_of_car = 2024 - int(model_year)

# Prediction Logic
def predict_price(fuel_type, body_type, transmission, kms_driven, owner_number, engine_displacement, max_power, torque_value, city, seating_capacity, insurance_validity, mileage, age_of_car):
    # Encode categorical variables
    fuel_type_encoded = fuel_type_mapping[fuel_type]
    body_type_encoded = body_type_mapping[body_type]
    transmission_encoded = transmission_mapping[transmission]
    insurance_validity_encoded = insurance_validity_mapping[insurance_validity]

    # Prepare input data
    input_data = {
        "Fuel_Type": fuel_type_encoded,
        "Body_Type": body_type_encoded,
        "Kms_Driven": kms_driven,
        "Transmission": transmission_encoded,
        "Owner_Number": owner_number,
        "Engine_Displacement": engine_displacement,
        "Max_Power": max_power,
        "Torque": torque_value,
        "Seating_Capacity": seating_capacity,
        "City": city,
        "Insurance_Validity": insurance_validity_encoded,
        "Mileage": mileage,
        "Age_of_Car": age_of_car
    }

    # Create DataFrame and perform one-hot encoding if needed
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    # Align the input data with the model's expected input features
    model_columns = model.feature_names_in_
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Make the prediction
    predicted_price = model.predict(input_df)

    return predicted_price[0]

# Auto-refreshing price prediction
predicted_price = predict_price(fuel_type, body_type, transmission, kms_driven, owner_number, engine_displacement, max_power, torque_value, city, seating_capacity, insurance_validity, mileage, age_of_car)

st.success(f"The estimated price for the car is ₹{predicted_price:,.2f}")
