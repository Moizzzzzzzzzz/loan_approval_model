import joblib
import os

try:
    columns = joblib.load('model/columns.pkl')
    print("Columns found:", columns)
    print("Type:", type(columns))
except Exception as e:
    print(f"Error: {e}")
