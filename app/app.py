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
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

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
    .stCard {
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
    st.markdown("1. **Enter Details**: Fill in the applicant's financial and personal information.")
    st.markdown("2. **Analyze**: Our model processes the data against historical trends.")
    st.markdown("3. **Result**: Get an instant approval probability.")
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.caption("Model: Logistic Regression")
    st.caption("Accuracy: ~80%")
    
    st.markdown("---")
    st.markdown("Made with ❤️ by Abdul")

# --- Main Content ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🏦 Loan Approval Prediction")
    st.markdown("### Fast, Accurate, and Secure Credit Assessment")
    st.markdown("Please provide the following details to check your loan eligibility.")

with col2:
    if lottie_bank:
        st_lottie(lottie_bank, height=150, key="bank")

st.markdown("---")

# --- Model Loading ---
@st.cache_resource
def load_artifacts():
    try:
        # Correct path: Go up one level from 'app' to project root, then into 'Model'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, '..', 'Model')
        
        # Load files
        model_path = os.path.join(model_dir, 'LogisticRegression.pkl')
        scaler_path = os.path.join(model_dir, 'StandardScaler.pkl')
        columns_path = os.path.join(model_dir, 'columns.pkl')

        if not all(os.path.exists(p) for p in [model_path, scaler_path, columns_path]):
             # Try lowercase 'model' just in case
             model_dir = os.path.join(base_dir, '..', 'model')
             model_path = os.path.join(model_dir, 'LogisticRegression.pkl')
             scaler_path = os.path.join(model_dir, 'StandardScaler.pkl')
             columns_path = os.path.join(model_dir, 'columns.pkl')
             
             if not all(os.path.exists(p) for p in [model_path, scaler_path, columns_path]):
                st.error(f"Model files not found in {model_dir} or ../Model")
                return None, None, None

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        columns = joblib.load(columns_path)

        return model, scaler, columns
        
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

# Load the model
model, scaler, feature_names = load_artifacts()

if model and scaler and feature_names is not None:
    
    # --- Input Form ---
    with st.container():
        st.subheader("📝 Applicant Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Personal Details")
            married = st.selectbox("Marital Status", ["No", "Yes"], help="Is the applicant married?")
            education = st.selectbox("Education Level", ["Not Graduate", "Graduate"], help="Highest level of education completed.")
            
        with col2:
            st.markdown("#### Financials")
            applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, step=100.0, value=5000.0, help="Monthly income of the applicant.")
            coapplicant_income = st.number_input("Co-Applicant Income ($)", min_value=0.0, step=100.0, value=0.0, help="Monthly income of the co-applicant.")
            
        with col3:
            st.markdown("#### Loan Details")
            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, step=100.0, value=150.0, help="Total loan amount requested.")
            property_area = st.selectbox("Property Area", ["Rural", "Urban", "Semi-Urban"], help="Location of the property.")
            credit_history = st.selectbox("Credit History", ["No", "Yes"], help="Does the applicant have a credit history (1.0) or not (0.0)?")

        st.markdown("---")
        
        # --- Prediction Logic ---
        if st.button("Predict Loan Approval", type="primary"):
            
            with st.spinner("Analyzing application..."):
                # Map inputs to model features
                # Features: ['Credit_History', 'Property_Area_Semiurban', 'Married', 'Education', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount']
                
                input_data = {
                    'Credit_History': 1.0 if credit_history == "Yes" else 0.0,
                    'Property_Area_Semiurban': 1 if property_area == "Semi-Urban" else 0,
                    'Married': 1 if married == "Yes" else 0,
                    'Education': 1 if education == "Graduate" else 0,
                    'Applicant_Income': applicant_income,
                    'Coapplicant_Income': coapplicant_income,
                    'Loan_Amount': loan_amount
                }
                
                # Create DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Ensure correct column order
                try:
                    input_df = input_df[feature_names]
                except KeyError as e:
                    st.error(f"Feature mismatch: {e}. Expected: {feature_names}")
                    st.stop()
                
                # Preprocess
                try:
                    scaled_data = scaler.transform(input_df)
                    
                    # Predict
                    prediction = model.predict(scaled_data)[0]
                    probability = model.predict_proba(scaled_data)[0][1]
                    
                    # --- Results ---
                    st.markdown("### Prediction Results")
                    
                    if prediction == 1: # Approved
                        st.balloons()
                        with st.container():
                            st.markdown(f"""
                                <div class="result-box approved">
                                    <h2>✅ Loan Approved</h2>
                                    <p style="font-size: 1.2rem;">Probability of Repayment: <strong>{probability:.2%}</strong></p>
                                    <p>Congratulations! Your application meets our criteria for approval.</p>
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
                                    <p>Based on the provided details, the application does not meet our current criteria.</p>
                                </div>
                            """, unsafe_allow_html=True)
                            if lottie_fail:
                                st_lottie(lottie_fail, height=200, key="fail")
                                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("⚠️ Model files could not be loaded. Please check the 'Model' directory and ensure all .pkl files are present.")
    st.info(f"Current working directory: {os.getcwd()}")
    st.info(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
