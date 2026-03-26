import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HealthGuard AI", layout="wide", page_icon="🩺")

# --- LOAD MODELS & ENCODERS ---
# Using the .pkl extensions as required for deployment [cite: 9, 18, 21]
try:
    breast_cancer_model = pickle.load(open('breast_cancer_xgb_model_revised.pkl', 'rb'))
    thyroid_model = pickle.load(open('Thyroid_disease_xgb_model_revised.pkl', 'rb'))
    thyroid_encoder = pickle.load(open('label_encoders_thyroid.pkl', 'rb'))
    heart_model = pickle.load(open('heart_disease_xgb_model.pkl', 'rb'))
    diabetes_model = pickle.load(open('diabetes.pkl', 'rb'))
    # autism_model = pickle.load(open('autism_model.pkl', 'rb')) #  Placeholder
except Exception as e:
    st.error(f"Error loading model files: {e}. Ensure all .pkl files are in the same folder.")

# --- NAVIGATION ---
with st.sidebar:
    selected = option_menu(
        "Health Prediction System",
        ["Breast Cancer", "Thyroid Disease", "Heart Disease", "Diabetes", "Autism"],
        icons=["gender-female", "activity", "heart-pulse", "droplet-half", "person-bounding-box"],
        menu_icon="hospital-fill",
        default_index=0,
    )

# --- MODULE 1: BREAST CANCER ---
if selected == "Breast Cancer":
    st.markdown("<h1 style='color: #d81b60;'>🎀 Breast Cancer Classification</h1>", unsafe_allow_html=True)
    st.info("Please enter the 30 clinical metrics for analysis [cite: 2-8].")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        v1 = st.number_input("radius_mean")
        v2 = st.number_input("texture_mean")
        v3 = st.number_input("perimeter_mean")
        v4 = st.number_input("area_mean")
        v5 = st.number_input("smoothness_mean")
        v6 = st.number_input("compactness_mean")
        v7 = st.number_input("concavity_mean")
        v8 = st.number_input("concave points_mean")
        v9 = st.number_input("symmetry_mean")
        v10 = st.number_input("fractal_dimension_mean")
    with col2:
        v11 = st.number_input("radius_se")
        v12 = st.number_input("texture_se")
        v13 = st.number_input("perimeter_se")
        v14 = st.number_input("area_se")
        v15 = st.number_input("smoothness_se")
        v16 = st.number_input("compactness_se")
        v17 = st.number_input("concavity_se")
        v18 = st.number_input("concave points_se")
        v19 = st.number_input("symmetry_se")
        v20 = st.number_input("fractal_dimension_se")
    with col3:
        v21 = st.number_input("radius_worst")
        v22 = st.number_input("texture_worst")
        v23 = st.number_input("perimeter_worst")
        v24 = st.number_input("area_worst")
        v25 = st.number_input("smoothness_worst")
        v26 = st.number_input("compactness_worst")
        v27 = st.number_input("concavity_worst")
        v28 = st.number_input("concave points_worst")
        v29 = st.number_input("symmetry_worst")
        v30 = st.number_input("fractal_dimension_worst")

    if st.button("Run Breast Cancer Analysis"):
        features = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30]
        prediction = breast_cancer_model.predict([features])
        result = "⚠️ Malignant" if prediction[0] == 1 else "✅ Benign"
        st.success(f"Prediction Result: {result}")

# --- MODULE 2: THYROID DISEASE (RESOLVED SHAPE MISMATCH) ---
elif selected == "Thyroid Disease":
    st.markdown("<h1 style='color: #6a1b9a;'>🦋 Thyroid Disease Prediction</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age_thy = st.number_input("Age", min_value=1, max_value=120) # Feature 1
        gender = st.selectbox("Gender", ["F", "M"])
        gender_map = {"F": 0, "M": 1} # [cite: 14]
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        smoking_map = {"No": 0, "Yes": 1} # [cite: 14]
        hx_smoking = st.selectbox("Hx Smoking", ["No", "Yes"])
        hx_smoking_map = {"No": 0, "Yes": 1} # [cite: 14]
        hx_radio = st.selectbox("Hx Radiothreapy", ["No", "Yes"])
        hx_radio_map = {"No": 0, "Yes": 1} # [cite: 14]
        thy_func = st.selectbox("Thyroid Function", ["Euthyroid", "Clinical Hyperthyroidism", "Subclinical Hypothyroidism", "Clinical Hypothyroidism", "Subclinical Hyperthyroidism"])
        thy_func_map = {"Clinical Hyperthyroidism": 0, "Clinical Hypothyroidism": 1, "Euthyroid": 2, "Subclinical Hyperthyroidism": 3, "Subclinical Hypothyroidism": 4} # [cite: 14]

    with col2:
        phys_exam = st.selectbox("Physical Examination", ["Multinodular goiter", "Single nodular goiter-right", "Single nodular goiter-left", "Normal", "Diffuse goiter"])
        phys_exam_map = {"Diffuse goiter": 0, "Multinodular goiter": 1, "Normal": 2, "Single nodular goiter-left": 3, "Single nodular goiter-right": 4} # [cite: 15]
        adenopathy = st.selectbox("Adenopathy", ["No", "Right", "Bilateral", "Left", "Extensive", "Posterior"])
        adenopathy_map = {"Bilateral": 0, "Extensive": 1, "Left": 2, "No": 3, "Posterior": 4, "Right": 5} # [cite: 15]
        pathology = st.selectbox("Pathology", ["Papillary", "Micropapillary", "Follicular", "Hurthel cell"])
        pathology_map = {"Follicular": 0, "Hurthel cell": 1, "Micropapillary": 2, "Papillary": 3} # [cite: 15]
        focality = st.selectbox("Focality", ["Uni-Focal", "Multi-Focal"])
        focality_map = {"Multi-Focal": 0, "Uni-Focal": 1} # [cite: 15]

    with col3:
        risk = st.selectbox("Risk", ["Low", "Intermediate", "High"])
        risk_map = {"High": 0, "Intermediate": 1, "Low": 2} # [cite: 16]
        t_stage = st.selectbox("T", ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
        t_map = {"T1a": 0, "T1b": 1, "T2": 2, "T3a": 3, "T3b": 4, "T4a": 5, "T4b": 6} # [cite: 16]
        n_stage = st.selectbox("N", ["N0", "N1a", "N1b"])
        n_map = {"N0": 0, "N1a": 1, "N1b": 2} # [cite: 16]
        m_stage = st.selectbox("M", ["M0", "M1"])
        m_map = {"M0": 0, "M1": 1} # [cite: 16]
        stage = st.selectbox("Stage", ["I", "II", "III", "IVA", "IVB"])
        stage_map = {"I": 0, "II": 1, "III": 2, "IVA": 3, "IVB": 4} # [cite: 16]
        response = st.selectbox("Response", ["Excellent", "Structural Incomplete", "Indeterminate", "Biochemical Incomplete"])
        response_map = {"Biochemical Incomplete": 0, "Excellent": 1, "Indeterminate": 2, "Structural Incomplete": 3} # 

    if st.button("Predict Thyroid Status"):
        # Features now includes Age (1) + Categoricals (14) + Response (1) = 16 Features
        features = [
            age_thy, gender_map[gender], smoking_map[smoking], hx_smoking_map[hx_smoking], 
            hx_radio_map[hx_radio], thy_func_map[thy_func], phys_exam_map[phys_exam],
            adenopathy_map[adenopathy], pathology_map[pathology], focality_map[focality],
            risk_map[risk], t_map[t_stage], n_map[n_stage], m_map[m_stage], stage_map[stage],
            response_map[response]
        ]
        prediction = thyroid_model.predict([features])
        result = "⚠️ Recurred" if prediction[0] == 1 else "✅ Not Recurred"
        st.success(f"Thyroid Recurrence Status: {result}")

# --- MODULE 3: HEART DISEASE ---
elif selected == "Heart Disease":
    st.markdown("<h1 style='color: #c62828;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
    st.info("Input patient metrics for heart disease screening [cite: 19-20].")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("age")
        sex = st.number_input("sex (1=Male, 0=Female)")
        cp = st.number_input("cp (Chest Pain Type)")
        trestbps = st.number_input("trestbps (Resting BP)")
        chol = st.number_input("chol")
        fbs = st.number_input("fbs")
        restecg = st.number_input("restecg")
    with col2:
        thalach = st.number_input("thalach")
        exang = st.number_input("exang")
        oldpeak = st.number_input("oldpeak")
        slope = st.number_input("slope")
        ca = st.number_input("ca")
        thal = st.number_input("thal")

    if st.button("Predict Heart Condition"):
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        prediction = heart_model.predict([features])
        result = "⚠️ Positive" if prediction[0] == 1 else "✅ Negative"
        st.success(f"Heart Disease Prediction: {result}")

# --- MODULE 4: DIABETES ---
elif selected == "Diabetes":
    st.markdown("<h1 style='color: #1565c0;'>🩸 Diabetes Prediction</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0) # [cite: 22]
        Glucose = st.number_input("Glucose") # [cite: 22]
        BloodPressure = st.number_input("BloodPressure") # [cite: 23]
        SkinThickness = st.number_input("SkinThickness") # [cite: 23]
    with col2:
        Insulin = st.number_input("Insulin") # [cite: 23]
        BMI = st.number_input("BMI") # [cite: 24]
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction") # [cite: 24]
        Age = st.number_input("Age", min_value=1) # [cite: 24]

    if st.button("Predict Diabetes Outcome"):
        features = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        prediction = diabetes_model.predict([features])
        result = "⚠️ Diabetic" if prediction[0] == 1 else "✅ Non-Diabetic"
        st.success(f"Result: {result}")
