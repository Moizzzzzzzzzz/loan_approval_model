import joblib
import pandas as pd
import os
import numpy as np

try:
    model_path = 'Model'
    print(f"Loading artifacts from {model_path}...")
    model = joblib.load(os.path.join(model_path, 'LogisticRegression.pkl'))
    scaler = joblib.load(os.path.join(model_path, 'StandardScaler.pkl'))
    columns = joblib.load(os.path.join(model_path, 'columns.pkl'))
    
    print("Columns:", columns)
    
    # Test Case 1: Default values (likely rejected due to Credit History=0)
    input_data_1 = {
        'Credit_History': 0.0,
        'Property_Area_Semiurban': 0,
        'Married': 0,
        'Education': 0,
        'Applicant_Income': 5000.0,
        'Coapplicant_Income': 0.0,
        'Loan_Amount': 150.0
    }
    
    # Test Case 2: Good values (Credit History=1, Graduate, Semiurban)
    input_data_2 = {
        'Credit_History': 1.0,
        'Property_Area_Semiurban': 1,
        'Married': 1,
        'Education': 1,
        'Applicant_Income': 5000.0,
        'Coapplicant_Income': 0.0,
        'Loan_Amount': 150.0
    }

    for i, data in enumerate([input_data_1, input_data_2]):
        print(f"\n--- Test Case {i+1} ---")
        df = pd.DataFrame([data])
        df = df[columns] # Ensure order
        print("Input Data:\n", df)
        
        scaled = scaler.transform(df)
        print("Scaled Data:\n", scaled)
        
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0]
        print(f"Prediction: {pred}")
        print(f"Probabilities: {proba}")

except Exception as e:
    print(f"Error: {e}")
