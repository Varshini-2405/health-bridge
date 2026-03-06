## HealthBridge: AI-Based Symptom Intelligence System

## The application will open at:
https://health-bridge.streamlit.app/


### HealthBridge is an AI-powered health triage system designed to improve healthcare accessibility in rural communities. The system analyzes patient symptoms using machine learning models to estimate health risk levels and predict possible diseases. It provides an easy-to-use interface for patients, a simulated IVR workflow for feature-phone users, and an admin dashboard for monitoring health trends.

## Key Features:

### 1.Patient Symptom Portal
Users can enter symptoms such as fever, cough, headache, and breathing difficulty. The system also considers age group as an input feature to improve prediction accuracy and provide risk assessment.

### 2.IVR Interaction Simulation
The platform includes an IVR simulation that demonstrates how feature-phone users could report symptoms using keypad inputs (DTMF). This ensures the system design remains inclusive for users without smartphones.

### 3.Admin Dashboard
The admin dashboard provides insights such as:
Total sessions and alerts
Risk distribution analysis
Detected disease cases
Most detected disease in the community
Logs of symptom reports and prediction confidence scores

## Machine Learning Predictions:
The system uses trained ML models to predict:
Risk Level: Low, Medium, High
Possible Disease: Based on symptom combinations
Predictions also include confidence scores to indicate model certainty.

## System Architecture:
HealthBridge runs as a unified Streamlit application.
Frontend: Streamlit interface with custom styling
Machine Learning Layer: Random Forest models for risk assessment and disease prediction
Data Logging: JSON-based storage for health interaction logs
Language Support: Kannada, English and Hindi

## Tech Stack:
Framework: Streamlit.
Machine Learning: Scikit-Learn, Pandas, NumPy, Joblib.
Visualization: Plotly.
UI Components: streamlit-option-menu.

## Installation:
1 Install dependencies
pip install -r requirements.txt

2 Run the application
streamlit run app_streamlit.py

## Machine Learning Models:
The system uses trained Random Forest models:

Model  |	Purpose
risk_model.pkl| Predicts health risk level
disease_model.pkl| Predicts possible disease
Model evaluation metrics include Accuracy, Precision, and Recall, which are displayed in the admin dashboard.

## Future Enhancements:
Real IVR integration using telephony APIs
Mobile application version
NLP-based free-text symptom input

Disease trend analysis for outbreak monitoring

