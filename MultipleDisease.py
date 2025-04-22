import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load models
with open("randomforest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("parkinsons_model.pkl", "rb") as f:
    pk_model = pickle.load(f)
with open("decisiontree_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

# Load scalers
with open("scaler_randomforest.pkl", "rb") as f:
    rf_scaler = pickle.load(f)
with open("scaler_decisiontree.pkl", "rb") as f:
    dt_scaler = pickle.load(f)
with open("scaler_parkinsons.pkl", "rb") as f:
    pk_scaler = pickle.load(f)

# Load encoders
with open("encoder_randomforest.pkl", "rb") as f:
    rf_encoder = pickle.load(f)
with open("encoder_decisiontree.pkl", "rb") as f:
    dt_encoder = pickle.load(f)
with open("encoder_parkinsons.pkl", "rb") as f:
    pk_encoder = pickle.load(f)

# Page config
st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")
st.sidebar.title("🔀 Navigation")
page = st.sidebar.radio("Choose Disease:", ["Liver Disease", "Kidney Disease", "Parkinsons Disease"])

# 1️⃣ Liver Disease
if page == "Liver Disease":
    st.title("Liver Disease Prediction")
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0)
    alkaline_phosphotase = st.number_input("Alkaline Phosphatose", min_value=0.0)
    alamine_aminotransferase = st.number_input("ALT", min_value=0.0)
    aspartate_aminotransferase = st.number_input("AST", min_value=0.0)

    if st.button("Predict"):
        input_data = pd.DataFrame([[age, total_bilirubin, alkaline_phosphotase,
                                    alamine_aminotransferase, aspartate_aminotransferase]],
                                  columns=["Age", "Total_Bilirubin", "Alkaline_Phosphotase",
                                           "Alamine_Aminotransferase", "Aspartate_Aminotransferase"])

        input_scaled = rf_scaler.transform(input_data)
        prediction = rf_model.predict(input_scaled)

        if prediction[0] == 0:
            st.success("✅ The patient is healthy.")
        else:
            st.error("⚠️ The patient has liver disease.")


# 2️⃣ Kidney Disease
elif page == "Kidney Disease":
    st.title("Chronic Kidney Disease Prediction")
    id_val = st.number_input("Patient ID", min_value=0, step=1)
    age = st.number_input("Age", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    sg = st.number_input("Specific Gravity", min_value=1.000, max_value=1.030, format="%.3f")
    al = st.number_input("Albumin", min_value=0)
    su = st.number_input("Sugar", min_value=0)
    rbc = st.selectbox("Red Blood Cells", options=["normal", "abnormal"])
    rbc_encoded = 1 if rbc == "normal" else 0

    if st.button("Predict"):
        input_data = pd.DataFrame([[id_val, age, bp, sg, al, su, rbc_encoded]],
                                  columns=["id", "age", "bp", "sg", "al", "su", "rbc"])

        input_scaled = dt_scaler.transform(input_data)
        prediction = dt_model.predict(input_scaled)
        if prediction == 0:
            st.success("✅ The patient is healthy.") 
        else:
            st.error("⚠️ The patient has kidney disease.")

# 3️⃣ Parkinson's Disease
elif page == "Parkinsons Disease":
    st.title("Parkinson's Disease Prediction")

    # Input fields (all numerical and relevant to model)
    #name = st.text_input("Name")  # Optional, but won't be used in prediction
    mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
    mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
    mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
    mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0)
    mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0)
    nhr = st.number_input("NHR", min_value=0.0)
    ppe = st.number_input("PPE", min_value=0.0)
    spread1 = st.number_input("Spread1", min_value=-10.0)
    spread2 = st.number_input("Spread2", min_value=-10.0)

    if st.button("Predict"):
        # Prepare the input data in the exact same order as during model training
        input_data = pd.DataFrame([[ppe, mdvp_fo, spread1, spread2,
                                    mdvp_rap, mdvp_fhi, mdvp_flo, mdvp_apq, nhr]],
                                  columns=['PPE','MDVP:Fo(Hz)','spread1','spread2',
                                           'MDVP:RAP','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:APQ','NHR'])


        # Scale the input data (make sure to apply the same scaling that was used during training)
        input_scaled = pk_scaler.transform(input_data)

        # Make prediction using the trained model
        prediction = pk_model.predict(input_scaled)

        # Show results based on the prediction
        if prediction[0] == 0:
            st.success("✅ The patient is healthy.")
        else:
            st.error("⚠️ The person has Parkinson's disease.")
