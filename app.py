import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

# Load the trained model and feature columns
model = joblib.load('price_prediction_model_updated.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom, #d3dfe8, #ffffff);  /* Gradient background */
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
            margin-top: -30px;  /* Adjust this value to control top spacing */
        }
        .description {
            font-size: 1.2em;
            color: #1C2833;
            text-align: center;
            padding: 5px 0 20px 0;
        }
        .centered-button {
            display: flex;
            justify-content: center;
            padding-top: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 1em;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='title'>Welcome to the Used Car Price Estimator</div>", unsafe_allow_html=True)
st.markdown("<div class='description'>Enter the details below to estimate the car price.</div>", unsafe_allow_html=True)

# Input section layout
col1, col2 = st.columns(2)

with col1:
    kms_driven = st.number_input("Kilometers Driven", min_value=1, max_value=500000, value=50000)
    model_year = st.number_input("Model Year", min_value=1980, max_value=2023, value=2020)
    seats = st.number_input("Number of Seats", min_value=1, max_value=10, value=5)
    mileage = st.number_input("Mileage (km/l)", min_value=1.0, max_value=50.0, value=15.0)
    engine_displacement = st.number_input("Engine Displacement (cc)", min_value=500, max_value=5000, value=1500)

with col2:
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'])
    body_type = st.selectbox("Body Type", [
        'Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 
        'Minivans', 'Pickup Trucks', 'Convertibles', 
        'Hybrids', 'Wagon'
    ])
    transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
    oem = st.selectbox("OEM", [
        'Maruti', 'Ford', 'Tata', 'Hyundai', 'Jeep', 'Datsun', 
        'Honda', 'Mahindra', 'Mercedes-Benz', 'BMW', 'Renault', 
        'Audi', 'Toyota', 'Mini', 'Kia', 'Skoda', 'Volkswagen', 
        'Volvo', 'MG', 'Nissan', 'Fiat', 'Mahindra Ssangyong', 
        'Mitsubishi', 'Jaguar', 'Land Rover', 'Chevrolet', 
        'Citroen', 'Opel', 'Mahindra Renault', 'Isuzu', 
        'Lexus', 'Porsche', 'Hindustan Motors'
    ])

# Initialize input data with zeros for compatibility with model feature names
input_data = pd.DataFrame(columns=feature_columns)
input_data.loc[0] = [0] * len(feature_columns)  # Initialize with zeros

# Function to validate input fields
def validate_input():
    # Check if any of the numeric fields are empty or invalid (i.e., <= 0)
    if kms_driven <= 0 or model_year <= 1980 or seats <= 0 or mileage <= 0 or engine_displacement <= 0:
        return False
    # Ensure no categorical field is left blank or invalid
    if not fuel_type or not body_type or not transmission or not oem:
        return False
    return True

# Fill in user input values
input_data.at[0, 'Kms Driven'] = kms_driven
input_data.at[0, 'Model Year'] = model_year
input_data.at[0, 'Seats'] = seats
input_data.at[0, 'Engine Displacement'] = engine_displacement
input_data.at[0, 'Mileage'] = mileage

# One-hot encoding for categorical features, only if columns exist in feature_columns
if f'Fuel Type_{fuel_type}' in input_data.columns:
    input_data.at[0, f'Fuel Type_{fuel_type}'] = 1
if f'Body Type_{body_type}' in input_data.columns:
    input_data.at[0, f'Body Type_{body_type}'] = 1
if f'Transmission_{transmission}' in input_data.columns:
    input_data.at[0, f'Transmission_{transmission}'] = 1
if f'OEM_{oem}' in input_data.columns:
    input_data.at[0, f'OEM_{oem}'] = 1

# Centered button with custom styling
st.markdown("<div class='centered-button'>", unsafe_allow_html=True)
predict_button = st.button("Estimate Car Price")
st.markdown("</div>", unsafe_allow_html=True)

# Handling errors: Check for empty or invalid input
if predict_button:
    # Validate the input before proceeding
    if not validate_input():
        # Show error message if input is incomplete
        st.error("⚠️ Please complete all fields before submitting.")
        st.warning("🧐 All fields are required to provide an estimate. Fill them out and try again!")
    else:
        # If validation passes, proceed with prediction

        # Create a placeholder for loading message
        loading_message = st.empty()

        # Show loading spinner while the prediction is being made and update message
        with st.spinner('🚗 Estimating car price...'):
            time.sleep(1)  # Simulate a small delay for the spinner
            loading_message.markdown("🚗 **We are analyzing your car's details...** Please wait...")
            time.sleep(1)
            loading_message.markdown("🎉 **Your estimated car price is ready!...**")
            time.sleep(1)  # Simulate a small delay before displaying the result

            try:
                # Generate prediction
                predicted_price = model.predict(input_data)[0]
            except Exception as e:
                # Handle prediction failure by showing an error message
                st.error("🚨 Something went wrong while estimating the car price. Please try again.")
                st.warning("Oops! We couldn't generate the estimate at the moment. Please try again later.")
                # Empty the loading message to stop the spinner
                loading_message.empty()
                # Don't display anything further if prediction fails
                st.stop()  # Stop further execution

        # Clear the loading message after the prediction is complete
        loading_message.empty()  # This will remove the loading message

        # Display the price in rupees format with commas
        formatted_price = f"₹{predicted_price:,.2f}"
        
        # Display the price next to the button
        st.markdown(f"<div style='text-align: center; font-size: 1.5em; font-weight: bold; color: #2E86C1;'>Estimated Price: {formatted_price}</div>", unsafe_allow_html=True)
        
        # Create a gauge chart to display the price
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_price,
            title={'text': "Estimated Price (₹)", 'font': {'size': 24}},
            number={'valueformat': ",", 'prefix': "₹", 'font': {'size': 32}},
            gauge={
                'axis': {'range': [0, max(2000000, predicted_price * 1.2)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "#d3dfe8",  # Light blue-gray background to match page's gradient
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, predicted_price * 0.5], 'color': '#d4f0f0'},
                    {'range': [predicted_price * 0.5, predicted_price], 'color': '#76c7c0'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_price
                }
            }
        ))

        fig.update_layout(
            height=400,  # Adjust gauge height
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Set paper background to transparent
            plot_bgcolor="#d3dfe8"  # Set plot background to light blue-gray
        )

        # Display the gauge chart in Streamlit
        st.plotly_chart(fig)
