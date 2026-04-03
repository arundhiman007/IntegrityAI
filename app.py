import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
from features import TextFeatureExtractor
from evaluator import ThinkingEvaluator

# -- Page Configuration --
st.set_page_config(page_title="EduGuard Core", layout="wide", page_icon="🛡️")

# -- Stunning Custom CSS --
st.markdown("""
<style>
    /* Dark glassmorphism theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .main-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.07);
        border-left: 4px solid #6366f1;
        padding: 1.5rem;
        border-radius: 12px;
        height: 100%;
    }
    div[data-baseweb="textarea"] > div {
        background-color: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
    }
    textarea {
        color: #e2e8f0 !important;
        font-size: 1.1rem !important;
    }
    h1, h2, h3, p {
        color: #e0e7ff;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    .title-gradient {
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        padding-bottom: 5px;
    }
    .subtitle {
        color: #94a3b8 !important;
        font-size: 1.25rem !important;
        font-weight: 400 !important;
    }
</style>
""", unsafe_allow_html=True)

# -- Cached Loaders --
@st.cache_resource(show_spinner=False)
def load_ml_components():
    model_path = "models/xgboost_ai_detector.pkl"
    clf = None
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    extractor = TextFeatureExtractor()
    evaluator = ThinkingEvaluator()
    return clf, extractor, evaluator

st.markdown("<div style='text-align: center;'><h1 class='title-gradient'>EduGuard Platform</h1></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;' class='subtitle'>Advanced academic integrity and critical thinking evaluator</p>", unsafe_allow_html=True)

with st.spinner("Initializing Deep Learning Engine..."):
    clf, extractor, evaluator = load_ml_components()

if clf is None:
    st.error("Model not found. Please run `python train.py` from the terminal first to generate the XGBoost detector.")
    st.stop()

# Layout main box
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("### 📝 Submission Engine")
text_input = st.text_area("Student Essay Submission", height=250, placeholder="Drop the student's work here for deep neural analysis...", label_visibility="collapsed")

if st.button("🚀 Run Deep Analysis", use_container_width=True, type="primary"):
    if len(text_input.strip()) < 30:
        st.warning("⚠️ Please enter a longer text sequence (at least a full sentence) for accurate NLP analysis.")
    else:
        st.markdown("---")
        
        with st.spinner("Extracting Transformer Embeddings & Stylometrics..."):
            stats = list(extractor.get_stylometric_features(text_input).values())
            emb = extractor.get_embeddings([text_input])[0]
            X = np.concatenate([stats, emb]).reshape(1, -1)
            
            prob_ai = float(clf.predict_proba(X)[0][1])
            is_ai = prob_ai > 0.5

        with st.spinner("Evaluating Logical Reasoning Rubric using LLM Agent..."):
            ct = evaluator.evaluate(text_input)

        st.markdown("## 🔍 Diagnostic Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### 🤖 Authorship Authenticity")
            if is_ai:
                st.error("🚨 **High Probability of AI Generation**")
            else:
                st.success("✅ **Likely Human Written**")
                
            st.metric(label="Calculated AI Probability", value=f"{prob_ai*100:.1f}%")
            st.progress(prob_ai)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### 🧠 Critical Thinking Rubric")
            if "error" in ct:
                st.error(f"Evaluation Error: {ct['error']}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Logic & Reasoning", f"{ct.get('logical_reasoning', 0)} / 10")
                c2.metric("Concept Quality", f"{ct.get('concept_quality', 0)} / 10")
                c3.metric("Originality", f"{ct.get('originality', 0)} / 10")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 Automated Actionable Feedback")
        if "feedback" in ct:
            st.info(ct["feedback"])
            
st.markdown("</div>", unsafe_allow_html=True)
