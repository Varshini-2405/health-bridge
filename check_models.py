import joblib
import os

risk_path = "ai-service/models/risk_model.pkl"
disease_path = "ai-service/models/disease_model.pkl"

def check_model(path):
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"Model: {path}")
        if hasattr(model, 'feature_names_in_'):
            print(f"Features: {model.feature_names_in_}")
        else:
            print("No feature names found (model might be old or not fitted with a DataFrame)")
    else:
        print(f"File not found: {path}")

check_model(risk_path)
check_model(disease_path)
