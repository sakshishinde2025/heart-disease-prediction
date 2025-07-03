import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load the Saved Model and Scaler ---
# We load the model and scaler objects that we saved in the model.py script.
# 'rb' stands for 'read binary', which is the mode for reading pickled files.
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Error: `model.pkl` or `scaler.pkl` not found. Please run `model.py` first to generate these files.")
    st.stop()


# --- 2. Set Up the Streamlit Page ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Add a title and a description
st.title("Heart Disease Prediction App ❤️")
st.write(
    "This application uses a machine learning model to predict whether a patient has heart disease "
    "based on their clinical data. Please enter the patient's information below."
)


# --- 3. Create the User Input Interface ---
st.sidebar.header("Patient Data Input")

# Create a function to hold the input fields in the sidebar
def user_input_features():
    st.sidebar.markdown("---")
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type', (1, 2, 3, 4))
    resting_bp_s = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
    fasting_blood_sugar = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('No', 'Yes'))
    resting_ecg = st.sidebar.selectbox('Resting ECG', (0, 1, 2))
    max_heart_rate = st.sidebar.slider('Max Heart Rate Achieved', 70, 220, 150)
    exercise_angina = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'))
    oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest (oldpeak)', 0.0, 6.2, 1.0)
    st_slope = st.sidebar.selectbox('Slope of the peak exercise ST segment', (1, 2, 3))

    # Convert categorical inputs to numerical format
    sex_num = 1 if sex == 'Male' else 0
    fasting_blood_sugar_num = 1 if fasting_blood_sugar == 'Yes' else 0
    exercise_angina_num = 1 if exercise_angina == 'Yes' else 0

    # Create a dictionary of the data
    data = {
        'age': age,
        'sex': sex_num,
        'chest pain type': chest_pain_type,
        'resting bp s': resting_bp_s,
        'cholesterol': cholesterol,
        'fasting blood sugar': fasting_blood_sugar_num,
        'resting ecg': resting_ecg,
        'max heart rate': max_heart_rate,
        'exercise angina': exercise_angina_num,
        'oldpeak': oldpeak,
        'ST slope': st_slope
    }
    
    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()


# --- 4. Display User Input and Make Predictions ---
st.subheader('Patient Data Summary')
st.write(input_df)

# When the 'Predict' button is clicked
if st.button('Predict'):
    # Scale the user input using the loaded scaler
    input_scaled = scaler.transform(input_df)
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Prediction Result')
    
    # Display the prediction
    if prediction[0] == 1:
        st.error('**Result: Heart Disease**')
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success('**Result: Normal**')
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")

    st.info("This prediction is based on a machine learning model and is not a substitute for professional medical advice. Please consult a doctor for any health concerns.")


# --- 5. Add Explanations ---
st.markdown("---")
with st.expander("About the Features"):
    st.write("""
    - **Age**: The patient's age in years.
    - **Sex**: The patient's sex (Male/Female).
    - **Chest Pain Type**: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic.
    - **Resting Blood Pressure**: The patient's resting blood pressure in mm Hg.
    - **Cholesterol**: The patient's serum cholesterol in mg/dl.
    - **Fasting Blood Sugar > 120 mg/dl**: Whether the patient's fasting blood sugar is greater than 120 mg/dl.
    - **Resting ECG**: Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy).
    - **Max Heart Rate Achieved**: The maximum heart rate achieved by the patient.
    - **Exercise Induced Angina**: Whether the patient experienced angina during exercise.
    - **Oldpeak**: ST depression induced by exercise relative to rest.
    - **ST Slope**: The slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping).
    """)
