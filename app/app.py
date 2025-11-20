import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from streamlit_lottie import st_lottie

# Page Config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Assets & Styles ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_bank = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_kuyc3y.json")
lottie_success = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_pqnfmone.json")
lottie_fail = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qpwbv5gm.json")

st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Card Style */
    .css-1r6slb0 {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button Style */
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #34495e;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Result Box */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 1s;
    }
    .approved {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .rejected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/bank-building.png")
    st.title("Loan Predictor")
    st.info("This AI-powered tool assesses loan eligibility with high accuracy using Logistic Regression.")
    st.markdown("---")
    st.markdown("### 🛠️ How it works")
    st.markdown("1. Enter applicant details.")
    st.markdown("2. Click 'Predict Loan Approval'.")
    st.markdown("3. Get instant results.")
    st.markdown("---")
    st.markdown("[View on GitHub](#)")

# --- Main Content ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🏦 Loan Approval Prediction")
    st.markdown("### Fast, Accurate, and Secure Credit Assessment")

with col2:
    if lottie_bank:
        st_lottie(lottie_bank, height=150, key="bank")

st.markdown("---")

# --- Model Loading ---
@st.cache_resource
def load_artifacts():
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model')
        model = joblib.load(os.path.join(model_path, 'LogisticRegression.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'StandardScaler.pkl'))
        columns = joblib.load(os.path.join(model_path, 'columns.pkl'))
        return model, scaler, columns
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, feature_names = load_artifacts()

if model and scaler and feature_names is not None:
    
    # --- Input Form ---
    with st.container():
        st.subheader("📝 Applicant Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            married = st.selectbox("Married", ["No", "Yes"])
            education = st.selectbox("Education", ["Not Graduate", "Graduate"])
            credit_history = st.selectbox("Credit History", ["No", "Yes"], help="Does the applicant have a credit history?")
            
        with col2:
            applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, step=100.0, value=5000.0)
            coapplicant_income = st.number_input("Co-Applicant Income ($)", min_value=0.0, step=100.0, value=0.0)
            
        with col3:
            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, step=100.0, value=150.0)
            property_area = st.selectbox("Property Area", ["Rural", "Urban", "Semi-Urban"])

        # --- Prediction Logic ---
        if st.button("Predict Loan Approval", type="primary"):
            
            # Map inputs to model features
            # Features: ['Credit_History', 'Property_Area_Semiurban', 'Married', 'Education', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount']
            
            input_data = {
                'Credit_History': 1.0 if credit_history == "Yes" else 0.0,
                'Property_Area_Semiurban': 1 if property_area == "Semi-Urban" else 0, # Assumption based on columns
                'Married': 1 if married == "Yes" else 0,
                'Education': 1 if education == "Graduate" else 0,
                'Applicant_Income': applicant_income,
                'Coapplicant_Income': coapplicant_income,
                'Loan_Amount': loan_amount
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure correct column order
            input_df = input_df[feature_names]
            
            # Preprocess
            try:
                scaled_data = scaler.transform(input_df)
                
                # Predict
                prediction = model.predict(scaled_data)[0]
                probability = model.predict_proba(scaled_data)[0][1]
                
                # --- Results ---
                st.markdown("---")
                
                if prediction == 1: # Approved
                    st.balloons()
                    with st.container():
                        st.markdown(f"""
                            <div class="result-box approved">
                                <h2>✅ Loan Approved</h2>
                                <p style="font-size: 1.2rem;">Probability of Repayment: <strong>{probability:.2%}</strong></p>
                                <p>Congratulations! Your application meets our criteria.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        if lottie_success:
                            st_lottie(lottie_success, height=200, key="success")
                else: # Rejected
                    with st.container():
                        st.markdown(f"""
                            <div class="result-box rejected">
                                <h2>❌ Loan Rejected</h2>
                                <p style="font-size: 1.2rem;">Probability of Repayment: <strong>{probability:.2%}</strong></p>
                                <p>Unfortunately, your application does not meet our criteria at this time.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        if lottie_fail:
                            st_lottie(lottie_fail, height=200, key="fail")
                            
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Model files not found. Please check the 'model' directory.")
