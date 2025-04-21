import streamlit as st
import pandas as pd
import pickle
import numpy as np
# load the model 
with open("randomforest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("parkinsons_model.pkl", "rb") as f:
    pk_model = pickle.load(f)

with open("decisiontree_model.pkl", "rb") as f:
    dt_model = pickle.load(f)
# Load scaler
with open("scaler_randomforest.pkl", "rb") as f:
    rf_scaler = pickle.load(f)
with open("scaler_decisiontree.pkl", "rb") as f:
    dt_scaler = pickle.load(f)
with open("scaler_parkinsons.pkl", "rb") as f:
    pk_scaler = pickle.load(f)
# Load label encoder
with open("encoder_randomforest.pkl", "rb") as f:
    rf_encoder = pickle.load(f)
with open("encoder_decisiontree.pkl", "rb") as f:
    dt_encoder = pickle.load(f)
with open("encoder_parkinsons.pkl", "rb") as f:
    pk_encoder = pickle.load(f)

# Page config
st.set_page_config(page_title="Multiple Disease prediction", layout="centered")

# Sidebar navigation
st.sidebar.title("🔀 Navigation")
page = st.sidebar.radio("Choose Disease:", ["Liver Disease", "Kidney Disease","Parkinsons Disease"])

if page == "Liver Disease":
    st.title(" Liver Disease prediction")
    # Input fields for the 5 features
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    total_bilirubin = st.number_input("Total Bilirubin (normal range = 0.1 – 1.2 mg/dL)", min_value=0)
    alkaline_phosphatase = st.number_input("Alkaline Phosphatase", min_value=0)
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT)", min_value=0)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST)", min_value=0)
    # Predict button
    if st.button("Predict"):
        input_data = pd.DataFrame([[age,total_bilirubin, alkaline_phosphatase,alamine_aminotransferase,aspartate_aminotransferase ]])
        
        # Scale input data
        #scaled_input = rf_scaler.transform(input_data)

        # Prediction
        prediction =rf_model.predict(input_data)

        # Result
        if prediction[0] == 1:
            st.error("⚠️ The patient has  liver disease.")
        else:
            st.success("✅ The patient is healthy.")
elif page == "Kidney Disease":
    st.title("Chronic Kidney Disease Prediction")


    # User inputs
    age = st.number_input("Age", min_value=0)
    bp = st.number_input("Blood Pressure (bp)(90 – 120 mm Hg )", min_value=0.0)
    bgr = st.number_input("Blood Glucose Random (bgr)(70 – 140 mg/dL)", min_value=0.0)
    bu = st.number_input("Blood Urea (bu)(7 – 20 mg/dL)", min_value=0.0)
    sc = st.number_input("Serum Creatinine (sc)(0.6 – 1.3 mg/dL)", min_value=0.0)
    sod = st.number_input("Sodium Level (sod)(135 – 145 mEq/L)", min_value=0.0)

    # Predict button
    if st.button("Predict"):
        # Prepare input array
        input_data = pd.DataFrame([[age, bp, bgr, bu, sc, sod]])

        # Scale the input
        #input_scaled = dt_scaler.transform(input_data)

        # Make prediction
        prediction = dt_model.predict(input_data)[0]

        # Result
        if prediction == 1:
            st.error("⚠️ The patient has  kidney disease.")
        else:
            st.success("✅ The patient is healthy.")
elif page == "Parkinsons Disease":
    st.title("Parkinson's Disease Prediction")
    

    # Input fields for the 10 features
    mdvp_shimmer = st.number_input("MDVP:Shimmer(0.01 – 0.05)", min_value=0.0)
    mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)(0.1 – 0.5)", min_value=0.0)
    mdvp_apq = st.number_input("MDVP:APQ(0.01 – 0.04)", min_value=0.0)
    shimmer_dda = st.number_input("Shimmer:DDA(0.005 – 0.02)", min_value=0.0)
    rpde = st.number_input("RPDE(0.3 – 0.7)", min_value=0.0)
    dfa = st.number_input("DFA(0.6 – 0.8)", min_value=0.0)
    spread1 = st.number_input("spread1(-6.0 – -2.0)", min_value=-10.0, max_value=0.0)
    spread2 = st.number_input("spread2(-4.0 – 0.0)", min_value=-10.0, max_value=0.0)
    d2 = st.number_input("D2(1.5 – 3.0)", min_value=0.0)
    ppe = st.number_input("PPE(0.1 – 0.5)", min_value=0.0)

    # Predict button
    if st.button("Predict"):
        # Combine inputs into a numpy array
        input_data = pd.DataFrame([[mdvp_shimmer, mdvp_shimmer_db, mdvp_apq, shimmer_dda,
                                rpde, dfa, spread1, spread2, d2, ppe]])
        
        # Scale input data
        #input_scaled = pk_scaler.transform(input_data)
        
        # Make prediction
        prediction = pk_model.predict(input_data)[0]
        
        # Output result
        if prediction == 1:
            st.error(" ⚠️ The person has Parkinson's disease.")
        else:
            st.success(" ✅ The patient is healthy.")