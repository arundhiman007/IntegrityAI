import joblib
import numpy as np
import os
from features import TextFeatureExtractor

def test_single_samples():
    model_path = "models/xgboost_ai_detector.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run train.py first.")
        return

    print("Loading XGBoost Model...")
    classifier = joblib.load(model_path)
    
    print("Loading Feature Extractor (DistilBERT)...")
    extractor = TextFeatureExtractor()
    
    # 1. A handcrafted statement simulating authentic human writing
    human_text = "Every time I look at how fast the climate is changing it genuinely worries me for the future. Like I remember the winters being so much colder when I was a kid, and now it barely snows here. We really need to do something about industrial pollution before it's too late!"
    
    # 2. A patterned, formal AI-like statement
    ai_text = "As an AI language model, I do not have personal feelings. However, climate change is widely recognized as a critical global challenge. Primarily, rising global temperatures result in significant disruptions to ecosystems, while secondarily causing extreme weather patterns."

    test_samples = [
        ("Human-esque Statement", human_text),
        ("AI-esque Statement", ai_text)
    ]
    
    print("\n" + "="*50)
    print("         REAL-TIME MODEL INFERENCE TEST")
    print("="*50)

    for label, text in test_samples:
        print(f"\nEvaluating: [{label}]")
        print(f"Text Snippet: '{text[:90]}...'")
        
        # Extract features (Stylometrics + Embeddings)
        stats = list(extractor.get_stylometric_features(text).values())
        emb = extractor.get_embeddings([text])[0]
        
        # Combine to 772-dimensional feature array (768 from BERT + 4 stats)
        X = np.concatenate([stats, emb]).reshape(1, -1)
        
        # Predict probabilities
        prob_ai = float(classifier.predict_proba(X)[0][1])
        prediction = "🤖 AI Generated" if prob_ai > 0.5 else "👨‍💻 Human Written"
        
        print(f"Prediction result => {prediction}")
        print(f"Confidence Level  => AI Probability: {prob_ai * 100:.2f} | Human Probability: {(1-prob_ai) * 100:.2f}%")

if __name__ == "__main__":
    test_single_samples()
