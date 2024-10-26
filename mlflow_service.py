import joblib
import pandas as pd
from model_loader import test_data

# Load the pre-trained model from joblib
model = joblib.load("model_CNC_Machine_Utilization_f1_1.joblib")

# Load test data
data = pd.read_csv("CNC_Machin_Utilization.csv", parse_dates=["Timestamp"])


test_data(data)
