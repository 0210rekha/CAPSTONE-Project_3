import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model
try:
    model = joblib.load('used_car_price_model.joblib')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")

# Load the data to get OEM options
@st.cache_data
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
fuel_type = st.selectbox("Fuel Type", list(fuel_type_mapping.keys()))
body_type = st.selectbox("Body Type", list(body_type_mapping.keys()))
kms_driven = st.number_input("Kms Driven", min_value=0, max_value=500000, step=500)
transmission = st.selectbox("Transmission", list(transmission_mapping.keys()))
owner_number = st.slider("Owner Number (Previous Owners)", 0, 5)

# Select OEM
oem = st.selectbox("OEM (Car Manufacturer)", oem_options)

# Filter models based on selected OEM
model_options = data[data['OEM'] == oem]['Model'].unique()
model_name = st.selectbox("Model", model_options)

# Filter variants based on selected model
variant_options = data[(data['OEM'] == oem) & (data['Model'] == model_name)]['Variant Name'].unique()
variant_name = st.selectbox("Variant Name", variant_options)

model_year = st.slider("Model Year", 1990, 2024)
engine_displacement = st.number_input("Engine Displacement (cc)", min_value=500, max_value=6000, step=100)
max_power = st.number_input("Max Power (bhp)", min_value=50, max_value=1000, step=10)
torque = st.text_input("Torque")

# Select City
city_options = data['City'].unique()
city = st.selectbox("City", city_options)

# Optional Features
if st.checkbox("Include Optional Features"):
    st.subheader("Optional Details")
    seating_capacity = st.number_input("Seating Capacity", min_value=2, max_value=10, step=1)
    insurance_validity = st.selectbox("Insurance Validity", list(insurance_validity_mapping.keys()))
    mileage = st.number_input("Mileage (kmpl)", min_value=5, max_value=50, step=1)
    age_of_car = 2024 - model_year
else:
    seating_capacity = 0  # Default to 0 if not included
    insurance_validity = "Valid"  # Default insurance validity
    mileage = 0  # Default to 0 if not included
    age_of_car = 2024 - model_year  # This can remain


# Prediction Button
if st.button("Predict Price"):
    try:
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
            "Torque": torque,
            "Seating_Capacity": seating_capacity,
            "City": city,
            "Insurance_Validity": insurance_validity_encoded,
            "Mileage": mileage,
            "Age_of_Car": age_of_car
        }

        # Create DataFrame and perform one-hot encoding if needed
        input_df = pd.DataFrame([input_data])
        input_df_1 = pd.get_dummies(input_df)

        # Align the input data with the model's expected input features
        model_columns = model.feature_names_in_  # Retrieve feature names from the model
        input_df_1 = input_df_1.reindex(columns=model_columns, fill_value=0)  # Ensure all features are present

        # Make the prediction
        predicted_price = model.predict(input_df_1)

        # Display the prediction
        st.success(f"The estimated price for the car is ₹{predicted_price[0]:,.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

