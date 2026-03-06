import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
import os
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="HealthBridge | AI Health Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- THEME & CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background: #0f1021 !important;
        color: #f8fafc !important;
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(217, 70, 239, 0.1) 0%, transparent 40%);
        z-index: -1;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 32px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2.5rem !important;
        border-radius: 16px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 10px 20px -10px rgba(139, 92, 246, 0.5) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 15px 30px -10px rgba(139, 92, 246, 0.7) !important;
    }

    .stCheckbox, .stSelectbox {
        background: rgba(255,255,255,0.02);
        padding: 0.5rem 1rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 0.3rem !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 24px;
    }

    header { visibility: hidden; }
    footer { visibility: hidden; }

    .premium-gradient-text {
        background: linear-gradient(135deg, #f8fafc 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }

    .phone-mockup {
        border: 8px solid #1e293b;
        border-radius: 40px;
        padding: 20px;
        background: #020617;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
    }
    
    .metric-box {
        border-left: 4px solid #8b5cf6;
        padding: 10px 20px;
        background: rgba(139, 92, 246, 0.05);
        margin-bottom: 10px;
        border-radius: 0 15px 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL_PATH = os.path.join(BASE_DIR, "ai-service", "models", "risk_model.pkl")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "ai-service", "models", "disease_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ai-service", "models", "metrics.json")
LOG_FILE = os.path.join(BASE_DIR, "server", "data", "logs.json")

# --- DATA & MODELS ---
def check_and_train_models():
    if not (os.path.exists(RISK_MODEL_PATH) and os.path.exists(DISEASE_MODEL_PATH)):
        import subprocess
        import sys
        with st.spinner("Initializing AI Brain Layer..."):
            train_script = os.path.join(BASE_DIR, "ai-service", "train_model.py")
            subprocess.run([sys.executable, train_script], check=True)

@st.cache_resource
def load_models_v2():
    check_and_train_models()
    try:
        risk = joblib.load(RISK_MODEL_PATH)
        disease = joblib.load(DISEASE_MODEL_PATH)
        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
        return risk, disease, metrics
    except Exception as e:
        st.error(f"Error loading models: {e}")
        # Force re-train if files are corrupted
        if os.path.exists(RISK_MODEL_PATH): os.remove(RISK_MODEL_PATH)
        if os.path.exists(DISEASE_MODEL_PATH): os.remove(DISEASE_MODEL_PATH)
        st.rerun()

risk_model, disease_model, model_metrics = load_models_v2()

def log_prediction(symptoms, prediction):
    log_entry = {"timestamp": datetime.now().isoformat(), "symptoms": symptoms, "prediction": prediction}
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try: logs = json.load(f)
            except: logs = []
    logs.append(log_entry)
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    with open(LOG_FILE, 'w') as f: json.dump(logs, f)

# --- LANGUAGE SELECTION ---
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'English'

# --- NAVIGATION ---
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<h1 style="margin:0; padding:0; font-size:2rem; font-weight:900;">Health<span style="color:#8b5cf6">Bridge</span></h1>', unsafe_allow_html=True)
with col2:
    st.session_state['lang'] = st.selectbox("Choose Language / ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ", ["English", "Hindi", "Kannada"], index=0 if st.session_state['lang'] == 'English' else (1 if st.session_state['lang'] == 'Hindi' else 2))

translations = {
    'English': {
        'portal': 'Patient Portal', 'admin': 'Admin Intelligence', 'ivr': 'IVR Simulation',
        'hero_title': 'AI-Based Symptom Intelligence System', 
        'hero_desc': 'Analyze patient symptoms using machine learning to estimate health risk and possible diseases.',
        'select_symp': 'Select Symptoms', 'analyze': 'Check Health Now', 'result': 'Prediction Result',
        'prob': 'Probable Condition', 'guidance': 'Recommended Action',
        'ivr_title': 'Voice IVR Simulation', 'ivr_desc': 'For users without smartphones (Feature Phones)',
        'ivr_step': 'Simulated Call Step', 'ivr_q': 'Question:', 'ivr_press': 'Press to Answer',
        'ivr_analysis': 'ML Prediction Result', 'ivr_conf': 'Prediction Confidence',
        'ivr_risk': 'Risk Level', 'ivr_cond': 'Estimated Condition',
        'risk_map': {0: "Low", 1: "Medium", 2: "High"},
        'conf_label': 'Prediction Confidence',
        'emergency': '🚨 EMERGENCY CARE: Visit the nearest District Hospital immediately.',
        'visit_phc': '⚠️ VISIT PHC: Contact your local Primary Health Centre for evaluation.',
        'monitor': '✅ MONITOR: Rest at home and monitor temperature for 24 hours.',
        'admin_title': 'Health Intelligence Console', 'model_perf': 'ML Model Performance Metrics',
        'total_sess': 'Total Sessions', 'crit_alerts': 'Critical Alerts', 'most_detected': 'Most Detected Disease',
        'condition_spikes': 'Detected Disease Cases by Type', 'risk_dist': 'Risk Distribution',
        'db_logs': 'Database Intelligence Logs',
        'ivr_connected': 'Connected to HealthBridge', 'ivr_no': '0 - No', 'ivr_yes': '1 - Yes',
        'ivr_q1': 'Do you have a Fever?', 'ivr_q2': 'Do you have a Cough?', 'ivr_q3': 'Shortness of Breath?',
        'ivr_analyzing': 'Analyzing Response...', 'ivr_hangup': 'Hang Up / Finish',
        'ivr_header': 'Interactive Voice Response System',
        'ivr_b1': '1. **Toll-Free Dialing**: Accessible from any basic phone via district numbers.',
        'ivr_b2': '2. **DTMF Keypad Response**: Non-smartphone users respond using physical keys (1/0).',
        'ivr_b3': '3. **Localized Audio**: Questions are voiced in regional dialects.',
        'ivr_b4': '4. **Centralized Monitoring**: Stores risk scores in the health surveillance database.',
        'disease_model_name': 'Disease Classification Model',
        'risk_model_name': 'Risk Assessment Model',
        'monitoring_idle': 'Monitoring idle.',
        'footer': '© 2026 HEALTHBRIDGE AI | RURAL HEALTH INTELLIGENCE SYSTEM',
        'disease_rec': {
            'Pneumonia': 'Visit Emergency immediately. Take antibiotics if prescribed. Use oxygen if needed.',
            'COVID-19': 'Isolate yourself. Monitor oxygen levels. Stay hydrated. Consult PHC.',
            'Malaria': 'Visit PHC for blood test. Complete the course of antimalarials. Use mosquito nets.',
            'Flu': 'Take rest and fluids. Use Paracetamol for fever. Monitor symptoms.',
            'Common Cold': 'Drink warm water. Take rest. Use salt water gargles.'
        }
    },
    'Hindi': {
        'portal': 'रोगी पोर्टल', 'admin': 'प्रशासनिक खुफिया', 'ivr': 'आईवीआर सिमुलेशन',
        'hero_title': 'एआई-आधारित लक्षण खुफिया प्रणाली', 
        'hero_desc': 'स्वास्थ्य जोखिम और संभावित बीमारियों का अनुमान लगाने के लिए मशीन लर्निंग का उपयोग करके रोगी के लक्षणों का विश्लेषण करें।',
        'select_symp': 'लक्षण चुनें', 'analyze': 'अभी स्वास्थ्य जांचें', 'result': 'विशलेषण परिणाम',
        'prob': 'संभावित स्थिति', 'guidance': 'अनुशंसित कार्रवाई',
        'ivr_title': 'आवाज आईवीआर सिमुलेशन', 'ivr_desc': 'बिना स्मार्टफोन वाले उपयोगकर्ताओं के लिए',
        'ivr_step': 'सिम्युलेटेड कॉल चरण', 'ivr_q': 'प्रश्न:', 'ivr_press': 'उत्तर देने के लिए दबाएं',
        'ivr_analysis': 'ML भविष्यवाणी परिणाम', 'ivr_conf': 'भವಿಷ್ಯवाणी का विश्वास',
        'ivr_risk': 'जोखिम स्तर', 'ivr_cond': 'अनुमानित स्थिति',
        'risk_map': {0: "कम", 1: "मध्यम", 2: "उच्च"},
        'conf_label': 'भविष्यवाणी का विश्वास',
        'emergency': '🚨 आपातकालीन देखभाल: तुरंत निकटतम जिला अस्पताल जाएं।',
        'visit_phc': '⚠️ PHC जाएं: मूल्यांकन के लिए अपने local प्राथमिक स्वास्थ्य केंद्र से संपर्क करें।',
        'monitor': '✅ निगरानी: घर पर आराम करें और 24 घंटे तापमान की निगरानी करें।',
        'admin_title': 'स्वास्थ्य खुफिया कंसोल', 'model_perf': 'ML मॉडल प्रदर्शन मेट्रिक्स',
        'total_sess': 'कुल सत्र', 'crit_alerts': 'गंभीर अलर्ट', 'most_detected': 'सर्वाधिक खोजी गई बीमारी',
        'condition_spikes': 'प्रकार के अनुसार खोजी गई बीमारी के मामले', 'risk_dist': 'जोखिम वितरण',
        'db_logs': 'डेटाबेस इंटेलिजेंस लॉग',
        'ivr_connected': 'हेल्थब्रिज से जुड़े', 'ivr_no': '0 - नहीं', 'ivr_yes': '1 - हाँ',
        'ivr_q1': 'क्या आपको बुखार है?', 'ivr_q2': 'क्या आपको खांसी है?', 'ivr_q3': 'क्या आपको सांस लेने में तकलीफ है?',
        'ivr_analyzing': 'प्रतिक्रिया का विश्लेषण कर रहे हैं...', 'ivr_hangup': 'कॉल समाप्त करें',
        'ivr_header': 'इंटरैक्टिव वॉयस रिस्पांस सिस्टम',
        'ivr_b1': '1. **टोल-फ्री डायलिंग**: जिला नंबरों के माध्यम से किसी भी बेसिक फोन से सुलभ।',
        'ivr_b2': '2. **DTMF कीपैड प्रतिक्रिया**: गैर-स्मार्टफोन उपयोगकर्ता भौतिक कुंजियों (1/0) का उपयोग करके उत्तर देते हैं।',
        'ivr_b3': '3. **स्थानीयकृत ऑडियो**: प्रश्न क्षेत्रीय बोलियों में बोले जाते हैं।',
        'ivr_b4': '4. **केंद्रीकृत निगरानी**: सार्वजनिक स्वास्थ्य निगरानी के लिए डेटाबेस में स्कोर संग्रहीत करता है।',
        'disease_model_name': 'रोग वर्गीकरण मॉडल',
        'risk_model_name': 'जोखिम मूल्यांकन मॉडल',
        'monitoring_idle': 'निगरानी निष्क्रिय है।',
        'footer': '© 2026 हेल्थब्रिज एआई | ग्रामीण स्वास्थ्य खुफिया प्रणाली',
        'disease_rec': {
            'Pneumonia': 'तुरंत इमरजेंसी वार्ड जाएं। यदि निर्धारित हो तो एंटीबायोटिक्स लें। जरूरत पड़ने पर ऑक्सीजन का उपयोग करें।',
            'COVID-19': 'खुद को अलग करें। ऑक्सीजन के स्तर की निगरानी करें। हाइड्रेटेड रहें। PHC से सलाह लें।',
            'Malaria': 'रक्त परीक्षण के लिए PHC जाएं। मलेरिया-रोधी दवाओं का कोर्स पूरा करें। मच्छरदानी का प्रयोग करें।',
            'Flu': 'आराम करें और तरल पदार्थ लें। बुखार के लिए पैरासिटामोल का प्रयोग करें। लक्षणों की निगरानी करें।',
            'Common Cold': 'गुनगुना पानी पिएं। आराम करें। नमक के पानी से गरारे करें।'
        }
    },
    'Kannada': {
        'portal': 'ರೋಗಿ ಪೋರ್ಟಲ್', 'admin': 'ಆಡಳಿತಾತ್ಮಕ ಬುದ್ಧಿವಂತಿಕೆ', 'ivr': 'IVR ಸಿಮ್ಯುಲೇಶನ್',
        'hero_title': 'AI-ಆಧಾರಿತ ರೋಗಲಕ್ಷಣ ಗುಪ್ತಚರ ವ್ಯವಸ್ಥೆ',
        'hero_desc': 'ಆರೋಗ್ಯದ ಅಪಾಯ ಮತ್ತು ಸಂಭವನೀಯ ಕಾಯಿಲೆಗಳನ್ನು ಅಂದಾಜು ಮಾಡಲು ಯಂತ್ರ ಕಲಿಕೆಯನ್ನು ಬಳಸಿಕೊಂಡು ರೋಗಿಯ ಲಕ್ಷಣಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಿ.',
        'select_symp': 'ಲಕ್ಷಣಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ', 'analyze': 'ಈಗ ಆರೋಗ್ಯ ತಪಾಸಣೆ ಮಾಡಿ', 'result': 'ವಿಶ್ಲೇಷಣೆ ಫಲಿತಾಂಶ',
        'prob': 'ಸಂಭವನೀಯ ಸ್ಥಿತಿ', 'guidance': 'ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ',
        'ivr_title': 'ಧ್ವನಿ IVR ಸಿಮ್ಯುಲೇಶನ್', 'ivr_desc': 'ಸ್ಮಾರ್ಟ್‌ಫೋನ್ ಇಲ್ಲದ ಬಳಕೆದಾರರಿಗಾಗಿ (ಫೀಚರ್ ಫೋನ್‌ಗಳು)',
        'ivr_step': 'ಸಿಮ್ಯುಲೇಟೆಡ್ ಕರೆ ಹಂತ', 'ivr_q': 'ಪ್ರಶ್ನೆ:', 'ivr_press': 'ಉತ್ತರಿಸಲು ಒತ್ತಿರಿ',
        'ivr_analysis': 'ML ಮುನ್ಸೂಚನೆ ಫಲಿತಾಂಶ', 'ivr_conf': 'ಮುನ್ಸೂಚನೆಯ ವಿಶ್ವಾಸ',
        'ivr_risk': 'ಅಪಾಯದ ಮಟ್ಟ', 'ivr_cond': 'ಅಂದಾಜು ಸ್ಥಿತಿ',
        'risk_map': {0: "ಕಡಿಮೆ", 1: "ಮಧ್ಯಮ", 2: "ಹೆಚ್ಚು"},
        'conf_label': 'ಮುನ್ಸೂಚನೆಯ ವಿಶ್ವಾಸ',
        'emergency': '🚨 ತುರ್ತು ಆರೈಕೆ: ತಕ್ಷಣ ಸಮೀಪದ ಜಿಲ್ಲಾ ಆಸ್ಪತ್ರೆಗೆ ಭೇಟಿ ನೀಡಿ.',
        'visit_phc': '⚠️ PHC ಗೆ ಭೇಟಿ ನೀಡಿ: ಮೌಲ್ಯಮಾಪನಕ್ಕಾಗಿ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಪ್ರಾಥಮಿಕ ಆರೋಗ್ಯ ಕೇಂದ್ರವನ್ನು ಸಂಪರ್ಕಿಸಿ.',
        'monitor': '✅ ಮೇಲ್ವಿಚಾರಣೆ: ಮನೆಯಲ್ಲಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ ಮತ್ತು 24 ಗಂಟೆಗಳ ಕಾಲ ತಾಪಮಾನವನ್ನು ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ.',
        'admin_title': 'ಆರೋಗ್ಯ ಗುಪ್ತಚರ ಕನ್ಸೋಲ್', 'model_perf': 'ML ಮಾಡೆಲ್ ಕಾರ್ಯಕ್ಷಮತೆಯ ಮೆಟ್ರಿಕ್ಸ್',
        'total_sess': 'ಒಟ್ಟು ಸೆಷನ್‌ಗಳು', 'crit_alerts': 'ನಿರ್ಣಾಯಕ ಎಚ್ಚರಿಕೆಗಳು', 'most_detected': 'ಹೆಚ್ಚು ಪತ್ತೆಯಾದ ಕಾಯಿಲೆ',
        'condition_spikes': 'ಪ್ರಕಾರದ ಪ್ರಕಾರ ಪತ್ತೆಯಾದ ಕಾಯಿಲೆ ಪ್ರಕರಣಗಳು', 'risk_dist': 'ಅಪಾಯದ ವಿತರಣೆ',
        'db_logs': 'ಡೇಟಾಬೇಸ್ ಇಂಟೆಲಿಜೆನ್ಸ್ ಲಾಗ್ಸ್',
        'ivr_connected': 'ಹೆಲ್ತ್‌ಬ್ರಿಡ್ಜ್‌ಗೆ ಸಂಪರ್ಕಿಸಲಾಗಿದೆ', 'ivr_no': '0 - ಇಲ್ಲ', 'ivr_yes': '1 - ಹೌದು',
        'ivr_q1': 'ನಿಮಗೆ ಜ್ವರವಿದೆಯೇ?', 'ivr_q2': 'ನಿಮಗೆ ಕೆಮ್ಮು ಇದೆಯೇ?', 'ivr_q3': 'ಉಸಿರಾಟದ ತೊಂದರೆ ಇದೆಯೇ?',
        'ivr_analyzing': 'ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...', 'ivr_hangup': 'ಕರೆಯನ್ನು ಮುಕ್ತಾಯಗೊಳಿಸಿ',
        'ivr_header': 'ಇಂಟರಾಕ್ಟಿವ್ ವಾಯ್ಸ್ ರೆಸ್ಪಾನ್ಸ್ ಸಿಸ್ಟಮ್',
        'ivr_b1': '1. **ಟೋಲ್-ಫ್ರೀ ಡಯಲಿಂಗ್**: ಜಿಲ್ಲಾ ಸಂಖ್ಯೆಗಳ ಮೂಲಕ ಯಾವುದೇ ಮೂಲಭೂತ ಫೋನ್‌ನಿಂದ ಪ್ರವೇಶಿಸಬಹುದು.',
        'ivr_b2': '2. **DTMF ಕೀಪ್ಯಾಡ್ ಪ್ರತಿಕ್ರಿಯೆ**: ಸ್ಮಾರ್ಟ್‌ಫೋನ್ ಅಲ್ಲದ ಬಳಕೆದಾರರು ಭೌತಿಕ ಕೀಗಳನ್ನು (1/0) ಬಳಸಿ ಪ್ರತಿಕ್ರಿಯಿಸುತ್ತಾರೆ.',
        'ivr_b3': '3. **ಸ್ಥಳೀಕರಿಸಿದ ಆಡಿಯೋ**: ಪ್ರಶ್ನೆಗಳನ್ನು ಪ್ರಾದೇಶಿಕ ಉಪಭಾಷೆಗಳಲ್ಲಿ ಧ್ವನಿಸಲಾಗುತ್ತದೆ.',
        'ivr_b4': '4. **ಕೇಂದ್ರೀಕೃತ ಮೇಲ್ವಿಚಾರಣೆ**: ಆರೋಗ್ಯ ಕಣ್ಗಾವಲು ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ಅಪಾಯದ ಅಂಕಗಳನ್ನು ಸಂಗ್ರಹಿಸುತ್ತದೆ.',
        'disease_model_name': 'ರೋಗ ವರ್ಗೀಕರಣ ಮಾದರಿ',
        'risk_model_name': 'ಅಪಾಯದ ಮೌಲ್ಯಮಾಪನ ಮಾದರಿ',
        'monitoring_idle': 'ಮಾನಿಟರಿಂಗ್ ಐಡಲ್ ಆಗಿದೆ.',
        'footer': '© 2026 ಹೆಲ್ತ್‌ಬ್ರಿಡ್ಜ್ ಎಐ | ಗ್ರಾಮೀಣ ಆರೋಗ್ಯ ಗುಪ್ತಚರ ವ್ಯವಸ್ಥೆ',
        'disease_rec': {
            'Pneumonia': 'ತಕ್ಷಣ ತುರ್ತು ಚಿಕಿತ್ಸಾ ವಿಭಾಗಕ್ಕೆ ಭೇಟಿ ನೀಡಿ. ವೈದ್ಯರು ಸೂಚಿಸಿದರೆ ಪ್ರತಿಜೀವಕಗಳನ್ನು (antibiotics) ತೆಗೆದುಕೊಳ್ಳಿ. ಅವಶ್ಯಕತೆಯಿದ್ದರೆ ಆಮ್ಲಜನಕ ಬಳಸಿ.',
            'COVID-19': 'ನಿಮ್ಮನ್ನು ನೀವು ಪ್ರತ್ಯೇಕಿಸಿಕೊಳ್ಳಿ (isolate). ಆಮ್ಲಜನಕದ ಮಟ್ಟವನ್ನು ಗಮನಿಸಿ. ಸಾಕಷ್ಟು ದ್ರವ ಆಹಾರ ಸೇವಿಸಿ. PHC ಸಂಪರ್ಕಿಸಿ.',
            'Malaria': 'ರಕ್ತ ಪರೀಕ್ಷೆಗಾಗಿ PHC ಗೆ ಭೇಟಿ ನೀಡಿ. ಮಲೇರಿಯಾ ವಿರೋಧಿ ಔಷಧಿಗಳ ಕೋರ್ಸ್ ಪೂರ್ಣಗೊಳಿಸಿ. ಸೊಳ್ಳೆ ಪರದೆ ಬಳಸಿ.',
            'Flu': 'ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ ಮತ್ತು ದ್ರವ ಆಹಾರ ಸೇವಿಸಿ. ಜ್ವರಕ್ಕೆ ಪ್ಯಾರಸಿಟಮಾಲ್ ಬಳಸಿ. ರೋಗಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.',
            'Common Cold': 'ಉಗುರು ಬೆಚ್ಚಗಿನ ನೀರು ಕುಡಿಯಿರಿ. ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ. ಉಪ್ಪು ನೀರಿನಿಂದ ಗಂಟಲು ಮುಕ್ಕಳಿಸಿ (gargle).'
        }
    }
}
T = translations[st.session_state['lang']]

selected = option_menu(
    menu_title=None,
    options=[T['portal'], T['ivr'], T['admin']],
    icons=["heart-pulse", "telephone-inbound", "layout-dashboard"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#a78bfa", "font-size": "16px"},
        "nav-link": {"font-size": "13px", "font-weight": "700", "text-transform": "uppercase", "letter-spacing": "1px", "color": "#94a3b8"},
        "nav-link-selected": {"background-color": "rgba(139, 92, 246, 0.1)", "border-bottom": "3px solid #8b5cf6", "color": "white"},
    }
)

st.markdown("---")

# --- PATIENT PORTAL (STARTING PAGE) ---
if selected == T['portal']:
    st.markdown(f'<h1 class="premium-gradient-text" style="font-size:3.5rem; text-align:center; margin-bottom:0;">{T["hero_title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:#94a3b8; font-size:1.1rem; margin-bottom:3rem;">{T["hero_desc"]}</p>', unsafe_allow_html=True)
    
    col_in, col_res = st.columns([1.5, 1])
    with col_in:
        st.markdown(f'<div class="glass-card"><h3>{T["select_symp"]}</h3>', unsafe_allow_html=True)
        
        # New Feature: Age Group
        age_label = "Select Age Group / ವಯಸ್ಸಿನ ಗುಂಪನ್ನು ಆಯ್ಕೆಮಾಡಿ" if st.session_state['lang'] == 'Kannada' else ("Select Age Group / आयु वर्ग चुनें" if st.session_state['lang'] == 'English' else "आयु वर्ग चुनें")
        age_opt = ["Child (0-12)", "Adult (13-59)", "Elderly (60+)"]
        age_group = st.selectbox(age_label, age_opt, index=1)
        age_map = {"Child (0-12)": 0, "Adult (13-59)": 1, "Elderly (60+)": 2}

        c1, c2 = st.columns(2)
        with c1:
            f1 = "Fever" if st.session_state['lang'] == 'English' else ("ಬುಖಾರ/ಜ್ವರ" if st.session_state['lang'] == 'Kannada' else "बुखार")
            f2 = "Cough" if st.session_state['lang'] == 'English' else ("ಕೆಮ್ಮು" if st.session_state['lang'] == 'Kannada' else "खांसी")
            f3 = "Fatigue" if st.session_state['lang'] == 'English' else ("ದಣಿವು" if st.session_state['lang'] == 'Kannada' else "थकान")
            fever = st.checkbox(f1)
            cough = st.checkbox(f2)
            fatigue = st.checkbox(f3)
        with c2:
            f4 = "Shortness of Breath" if st.session_state['lang'] == 'English' else ("ಉಸಿರಾಟದ ತೊಂದರೆ" if st.session_state['lang'] == 'Kannada' else "सांस की तकलीफ")
            f5 = "Headache" if st.session_state['lang'] == 'English' else ("ತಲೆನೋವು" if st.session_state['lang'] == 'Kannada' else "सिरदर्द")
            f6 = "Body Ache" if st.session_state['lang'] == 'English' else ("ಅಂಗಾಂಗ ನೋವು" if st.session_state['lang'] == 'Kannada' else "बदन दर्द")
            sob = st.checkbox(f4)
            headache = st.checkbox(f5)
            body = st.checkbox(f6)
        
        symptoms = {'fever': fever, 'cough': cough, 'fatigue': fatigue, 'shortness_of_breath': sob, 'headache': headache, 'body_ache': body, 'sore_throat': False, 'age_group': age_map[age_group]}
        
        if st.button(T['analyze']):
            input_data = pd.DataFrame([symptoms])
            
            # Predict and Confidences
            risk_probs = risk_model.predict_proba(input_data)[0]
            risk_idx = np.argmax(risk_probs)
            risk_conf = round(risk_probs[risk_idx] * 100, 1)
            risk_level = T['risk_map'][risk_idx]
            
            dis_probs = disease_model.predict_proba(input_data)[0]
            dis_idx = np.argmax(dis_probs)
            dis_conf = round(dis_probs[dis_idx] * 100, 1)
            disease = disease_model.classes_[dis_idx] # Keep disease name in English for DB, translate on display if needed
            
            st.session_state['res'] = {
                "risk": risk_level, 
                "disease": disease, 
                "confidence": dis_conf,
                "symptoms": symptoms,
                "risk_idx": risk_idx
            }
            log_prediction(symptoms, {"risk_level": {0:"Low",1:"Medium",2:"High"}[risk_idx], "probable_disease": disease, "confidence": dis_conf})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        if 'res' in st.session_state:
            res = st.session_state['res']
            color = "#ef4444" if res['risk_idx'] == 2 else "#f59e0b" if res['risk_idx'] == 1 else "#10b981"
            st.markdown(f'<div style="background:{color}10; padding:2.5rem; border-radius:32px; border:1px solid {color}30; text-align:center;">', unsafe_allow_html=True)
            st.markdown(f'<h2 style="color:{color}; font-size:3.5rem; font-weight:900; margin:0.5rem 0;">{res["risk"]}</h2>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#94a3b8; font-size:0.8rem; margin:0;">{T["conf_label"]}: {res["confidence"]}%</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#94a3b8; margin:10px 0 0 0;">{T["prob"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:1.5rem; font-weight:800; color:#d946ef;">{res["disease"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f"### {T['guidance']}")
            if res['risk_idx'] == 2: st.error(T['emergency'])
            elif res['risk_idx'] == 1: st.warning(T['visit_phc'])
            else: st.success(T['monitor'])
            
            # Disease-specific recommendation
            st.info(f"💡 **{res['disease']} {T['guidance']}:** {T['disease_rec'].get(res['disease'], '')}")

# --- IVR SIMULATION ---
elif selected == T['ivr']:
    st.markdown(f'<h1 class="premium-gradient-text" style="font-size:3rem;">{T["ivr_title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#94a3b8;">{T["ivr_desc"]}</p>', unsafe_allow_html=True)
    
    col_p, col_i = st.columns([1.2, 1.8])
    with col_p:
        st.markdown('<div class="phone-mockup">', unsafe_allow_html=True)
        st.markdown('<div style="background:#1e293b; height:10px; width:40%; margin:0 auto 20px; border-radius:10px;"></div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#a78bfa; text-align:center; font-style:italic; font-size:14px; margin-bottom:20px;">🔊 "{T["ivr_connected"]}"</p>', unsafe_allow_html=True)
        
        if 'ivr_step' not in st.session_state: st.session_state['ivr_step'] = 1
        if 'ivr_data' not in st.session_state: st.session_state['ivr_data'] = {'age_group': 1}

        label_no = T['ivr_no']
        label_yes = T['ivr_yes']

        if st.session_state['ivr_step'] == 1:
            st.write(f"**🔊 {T['ivr_step']} 1:** {T['ivr_q1']}")
            c1, c2 = st.columns(2)
            if c1.button(label_no, key="ivr_f_0"): st.session_state['ivr_data']['fever'] = 0; st.session_state['ivr_step'] = 2; st.rerun()
            if c2.button(label_yes, key="ivr_f_1"): st.session_state['ivr_data']['fever'] = 1; st.session_state['ivr_step'] = 2; st.rerun()
            
        elif st.session_state['ivr_step'] == 2:
            st.write(f"**🔊 {T['ivr_step']} 2:** {T['ivr_q2']}")
            c1, c2 = st.columns(2)
            if c1.button(label_no, key="ivr_c_0"): st.session_state['ivr_data']['cough'] = 0; st.session_state['ivr_step'] = 3; st.rerun()
            if c2.button(label_yes, key="ivr_c_1"): st.session_state['ivr_data']['cough'] = 1; st.session_state['ivr_step'] = 3; st.rerun()

        elif st.session_state['ivr_step'] == 3:
            st.write(f"**🔊 {T['ivr_step']} 3:** {T['ivr_q3']}")
            c1, c2 = st.columns(2)
            if c1.button(label_no, key="ivr_s_0"): st.session_state['ivr_data']['shortness_of_breath'] = 0; st.session_state['ivr_step'] = 4; st.rerun()
            if c2.button(label_yes, key="ivr_s_1"): st.session_state['ivr_data']['shortness_of_breath'] = 1; st.session_state['ivr_step'] = 4; st.rerun()

        elif st.session_state['ivr_step'] == 4:
            st.write(f"**🔊 {T['ivr_analyzing']}**")
            s_ivr = {**{'fever':0, 'cough':0, 'fatigue':0, 'shortness_of_breath':0, 'headache':0, 'body_ache':0, 'sore_throat':0, 'age_group': 1}, **st.session_state['ivr_data']}
            input_ivr = pd.DataFrame([s_ivr])
            
            probs = disease_model.predict_proba(input_ivr)[0]
            max_idx = np.argmax(probs)
            conf = round(probs[max_idx] * 100, 1)
            risk_idx = np.argmax(risk_model.predict_proba(input_ivr)[0])
            risk_ivr = T['risk_map'][risk_idx]
            disease_ivr = disease_model.classes_[max_idx]
            
            st.session_state['ivr_res'] = {"risk": risk_ivr, "disease": disease_ivr, "confidence": conf}
            log_prediction(s_ivr, {"risk_level": {0:"Low",1:"Medium",2:"High"}[risk_idx], "probable_disease": disease_ivr, "confidence": conf})
            
            if st.button(T['ivr_hangup']): 
                st.session_state['ivr_step'] = 1; st.session_state['ivr_data'] = {}; st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col_i:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(T['ivr_header'])
        st.write(f"""
        {T['ivr_b1']}
        {T['ivr_b2']}
        {T['ivr_b3']}
        {T['ivr_b4']}
        """)
        
        if 'ivr_res' in st.session_state:
            ir = st.session_state['ivr_res']
            st.markdown("---")
            st.markdown(f"""
                <div style="background:rgba(139, 92, 246, 0.1); padding:1.5rem; border-radius:20px; border:1px solid rgba(139, 92, 246, 0.2);">
                    <h5 style="color:#a78bfa; margin-top:0;">🔊 {T['ivr_analysis']}</h5>
                    <p style="font-size:1rem; margin-bottom:5px;">{T['ivr_risk']}: <b>{ir['risk']}</b></p>
                    <p style="font-size:1rem; margin-bottom:5px;">{T['ivr_cond']}: <b>{ir['disease']}</b></p>
                    <p style="font-size:0.9rem; color:#d1d5db; margin-top:10px;">📋 {T['disease_rec'].get(ir['disease'], '')}</p>
                    <p style="font-size:0.8rem; color:#64748b; margin-top:10px;">{T['ivr_conf']}: {ir['confidence']}%</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- ADMIN DASHBOARD ---
elif selected == T['admin']:
    st.markdown(f'<h1 class="premium-gradient-text" style="font-size:3rem;">{T["admin_title"]}</h1>', unsafe_allow_html=True)
    
    # ML Model Performance Box
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(T['model_perf'])
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown(f"**{T['disease_model_name']}**")
        d_m = model_metrics.get('disease', {})
        st.markdown(f"<div class='metric-box'>Accuracy: {d_m.get('accuracy',0)} | Precision: {d_m.get('precision',0)} | Recall: {d_m.get('recall',0)}</div>", unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"**{T['risk_model_name']}**")
        r_m = model_metrics.get('risk', {})
        st.markdown(f"<div class='metric-box'>Accuracy: {r_m.get('accuracy',0)} | Precision: {r_m.get('precision',0)} | Recall: {r_m.get('recall',0)}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f: logs = json.load(f)
        df = pd.DataFrame(logs)
        df['risk_level'] = df['prediction'].apply(lambda x: x['risk_level'])
        df['disease'] = df['prediction'].apply(lambda x: x['probable_disease'])
        df['confidence'] = df['prediction'].apply(lambda x: x.get('confidence', '--'))
        
        m1, m2, m3 = st.columns(3)
        m1.metric(T['total_sess'], len(df))
        m2.metric(T['crit_alerts'], len(df[df['risk_level'] == 'High']))
        m3.metric(T['most_detected'], df['disease'].mode()[0] if not df.empty else "--")
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(T['condition_spikes'])
            cdf = df['disease'].value_counts().reset_index(); cdf.columns = ['disease', 'count']
            fig = px.bar(cdf, x='disease', y='count', color_discrete_sequence=['#8b5cf6'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(T['risk_dist'])
            fig_p = px.pie(df, names='risk_level', color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'}, hole=0.5)
            fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e2e8f0")
            st.plotly_chart(fig_p, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader(T['db_logs'])
        st.dataframe(df[['timestamp', 'risk_level', 'disease', 'confidence']].sort_values('timestamp', ascending=False), use_container_width=True)
    else: st.info(T['monitoring_idle'])

st.markdown(f'<p style="text-align:center; opacity:0.3; margin-top:5rem; font-size:10px; font-weight:800; letter-spacing:5px;">{T["footer"]}</p>', unsafe_allow_html=True)
