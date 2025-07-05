import streamlit as st
import requests
import pickle

# Load label encoder to get valid states
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
valid_states = le.classes_.tolist()

st.title("Suicide Rate Prediction")

# Input form
state = st.selectbox("State", valid_states)
number_of_suicides = st.number_input("Number of Suicides", min_value=0, value=0, step=1)
percentage_share = st.number_input("Percentage Share in Total Suicides", min_value=0.0, value=0.0, step=0.1)
population = st.number_input("Projected Mid-year Population (in lakhs)", min_value=0.0, value=0.0, step=0.01)
total_2020 = st.number_input("2020 Total", min_value=0, value=0, step=1)
percentage_variation = st.number_input("Percentage Variation", value=0.0, step=0.1)
male = st.number_input("Male Suicides", min_value=0, value=0, step=1)
female = st.number_input("Female Suicides", min_value=0, value=0, step=1)
transgender = st.number_input("Transgender Suicides", min_value=0, value=0, step=1)
year = st.number_input("Year", min_value=1990, max_value=2025, value=2020, step=1)

if st.button("Predict"):
    # Prepare data for API
    input_data = {
        "State": state,
        "Number_of_Suicides": number_of_suicides,
        "Percentage_share_in_total_suicides": percentage_share,
        "Projected_mid_year_population_in_lakhs": population,
        "Total_2020": total_2020,
        "Percentage_variation": percentage_variation,
        "Male": male,
        "Female": female,
        "Transgender": transgender,
        "Year": year
    }
    
    # Make API call
    try:
        response = requests.post("http://localhost:8000/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Predicted Suicide Rate: {result['predicted_suicide_rate']:.2f}")
        else:
            st.error(f"API call failed with status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Please ensure the FastAPI server is running.")