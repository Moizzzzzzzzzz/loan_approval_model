import joblib
import os
import numpy as np

try:
    scaler = joblib.load('model/StandardScaler.pkl')
    print("Scaler Mean:", scaler.mean_)
    print("Scaler Scale:", scaler.scale_)
    print("Feature Names (implied order):", ['Credit_History', 'Property_Area_Semiurban', 'Married', 'Education', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount'])
except Exception as e:
    print(f"Error: {e}")
