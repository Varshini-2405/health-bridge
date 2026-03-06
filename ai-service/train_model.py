import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os
import json

# Create data directory relative to this script
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "ai-service", "data")
MODELS_DIR = os.path.join(BASE_DIR, "ai-service", "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Generate Synthetic Health Data
# Symptoms: fever, cough, fatigue, shortness_of_breath, headache, body_ache, sore_throat
# Diseases: Flu, COVID-19, Common Cold, Pneumonia, Malaria
# Risk Levels: 0 (Low), 1 (Medium), 2 (High)

def generate_data(n_samples=2500):
    np.random.seed(42)
    symptoms = ['fever', 'cough', 'fatigue', 'shortness_of_breath', 'headache', 'body_ache', 'sore_throat']
    
    data = []
    for _ in range(n_samples):
        # Base probabilities for symptoms
        row = {s: np.random.choice([0, 1], p=[0.7, 0.3]) for s in symptoms}
        
        # New Feature: Age Group (0: Child, 1: Adult, 2: Elderly)
        age_group = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        row['age_group'] = age_group
        
        # Logic for disease and risk
        fever = row['fever']
        sob = row['shortness_of_breath']
        cough = row['cough']
        
        # Vulnerabilities based on age
        risk_multiplier = 1.2 if age_group in [0, 2] else 1.0
        
        if sob == 1 and fever == 1:
            disease = 'Pneumonia'
            risk = 2 # High
        elif sob == 1:
            disease = 'COVID-19'
            risk = 2 # High
        elif fever == 1 and row['body_ache'] == 1:
            disease = 'Malaria'
            risk = 1 # Medium
        elif fever == 1 and cough == 1:
            disease = 'Flu'
            risk = 1 # Medium
        else:
            disease = 'Common Cold'
            risk = 0 # Low
            
        row['disease'] = disease
        row['risk_level'] = risk
        data.append(row)
        
    return pd.DataFrame(data)

df = generate_data()
csv_path = os.path.join(DATA_DIR, "health_data.csv")
df.to_csv(csv_path, index=False)
print(f"Dataset generated successfully at {csv_path}")

# 2. Train Models
X = df.drop(['disease', 'risk_level'], axis=1)
y_risk = df['risk_level']
y_disease = df['disease']

# Model metrics to be saved for Admin dashboard
metrics = {}

# Risk Model
X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train, y_train)
risk_model_path = os.path.join(MODELS_DIR, "risk_model.pkl")
joblib.dump(risk_model, risk_model_path)

# Metrics for Risk Model
y_pred = risk_model.predict(X_test)
metrics['risk'] = {
    'accuracy': round(accuracy_score(y_test, y_pred), 2),
    'precision': round(precision_score(y_test, y_pred, average='weighted'), 2),
    'recall': round(recall_score(y_test, y_pred, average='weighted'), 2)
}

# Disease Model
X_train, X_test, y_train, y_test = train_test_split(X, y_disease, test_size=0.2, random_state=42)
disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
disease_model.fit(X_train, y_train)
disease_model_path = os.path.join(MODELS_DIR, "disease_model.pkl")
joblib.dump(disease_model, disease_model_path)

# Metrics for Disease Model
y_pred = disease_model.predict(X_test)
metrics['disease'] = {
    'accuracy': round(accuracy_score(y_test, y_pred), 2),
    'precision': round(precision_score(y_test, y_pred, average='weighted'), 2),
    'recall': round(recall_score(y_test, y_pred, average='weighted'), 2)
}

# Save metrics to a JSON file for the Streamlit UI to display
metrics_path = os.path.join(MODELS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f)

print(f"Models trained. Metrics saved to {metrics_path}.")
