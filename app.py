import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json


st.set_page_config(
    page_title="Student Mental Health Detector",
    layout="centered"
)


@st.cache_resource
def load_model():
    model = joblib.load('rf_model.pkl')
    return model

model = load_model()


st.title(" Student Mental Health & Burnout Detector")
st.markdown("""
**BCI3333 — MLA Final Assesment BY CA23108 FARIS WAZIRUL KENCHANA **  
Smart Campus | Student Well-being Early Detection System  
*Predict depression risk based on academic and lifestyle factors.*
""")
st.divider()

st.subheader(" Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    gender         = st.selectbox("Gender", ["Male", "Female"])
    age            = st.slider("Age", 17, 35, 21)
    cgpa           = st.slider("CGPA", 0.0, 4.0, 3.0, step=0.01)
    academic_press = st.slider("Academic Pressure (1–5)", 1, 5, 3)
    work_press     = st.slider("Work Pressure (1–5)", 1, 5, 2)

with col2:
    study_sat      = st.slider("Study Satisfaction (1–5)", 1, 5, 3)
    job_sat        = st.slider("Job Satisfaction (1–5)", 1, 5, 3)
    work_hrs       = st.slider("Work/Study Hours per day", 1, 12, 6)
    fin_stress     = st.slider("Financial Stress (1–5)", 1, 5, 2)
    sleep          = st.selectbox("Sleep Duration", [
        "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])

st.divider()
col3, col4 = st.columns(2)
with col3:
    dietary        = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    degree         = st.selectbox("Degree", ["B.Tech", "BA", "BCA", "BSc", "MBA", "M.Tech", "Others"])
with col4:
    suicidal       = st.selectbox("History of suicidal thoughts?", ["No", "Yes"])
    family_hist    = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])

st.divider()

sleep_map    = {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3}
dietary_map  = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
gender_map   = {"Female": 0, "Male": 1}
binary_map   = {"No": 0, "Yes": 1}
degree_map   = {"B.Tech": 0, "BA": 1, "BCA": 2, "BSc": 3, "MBA": 4, "M.Tech": 5, "Others": 6}

input_data = pd.DataFrame([{
    'Gender'                              : gender_map[gender],
    'Age'                                 : age,
    'Academic Pressure'                   : academic_press,
    'Work Pressure'                       : work_press,
    'CGPA'                                : cgpa,
    'Study Satisfaction'                  : study_sat,
    'Job Satisfaction'                    : job_sat,
    'Sleep Duration'                      : sleep_map[sleep],
    'Dietary Habits'                      : dietary_map[dietary],
    'Degree'                              : degree_map[degree],
    'Have you ever had suicidal thoughts ?': binary_map[suicidal],
    'Work/Study Hours'                    : work_hrs,
    'Financial Stress'                    : fin_stress,
    'Family History of Mental Illness'    : binary_map[family_hist]
}])


if st.button("🔍 Predict Depression Risk", use_container_width=True, type="primary"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    risk_pct = probability[1] * 100

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Depression Detected**")
        st.metric("Depression Risk Score", f"{risk_pct:.1f}%", delta="Above threshold")
        st.warning("""
        **Recommendation:** This student shows indicators associated with depression.  
        Early intervention is advised. Please consult a campus counsellor or mental health professional.
        """)
    else:
        st.success(f"✅ **Low Risk of Depression**")
        st.metric("Depression Risk Score", f"{risk_pct:.1f}%", delta="Below threshold")
        st.info("The student's current indicators suggest low depression risk. "
                "Continue monitoring academic and lifestyle factors.")

   
    st.progress(int(risk_pct))

    st.subheader("🔑 Key Risk Factors (from your input)")
    factors = {
        "Academic Pressure": academic_press / 5,
        "Financial Stress" : fin_stress / 5,
        "Work Pressure"    : work_press / 5,
        "Poor Sleep"       : 1 - sleep_map[sleep] / 3,
        "Unhealthy Diet"   : dietary_map[dietary] / 2,
    }
    for factor, score in sorted(factors.items(), key=lambda x: -x[1]):
        st.write(f"**{factor}**")
        st.progress(score)

st.divider()

