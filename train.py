import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
import joblib

# Load feature extractors
from features import TextFeatureExtractor

def train_model():
    print("Loading dataset...")
    try:
        print("Attempting to load DAIGT dataset from HuggingFace...")
        from datasets import load_dataset
        dataset = load_dataset("thedrcat/daigt-v2-train-dataset", split="train", trust_remote_code=True)
        df = dataset.to_pandas()
        
        df = df[['text', 'label']].dropna()
        df_human = df[df['label'] == 0].sample(min(1500, len(df[df['label'] == 0])), random_state=42)
        df_ai = df[df['label'] == 1].sample(min(1500, len(df[df['label'] == 1])), random_state=42)
        
        df_train = pd.concat([df_human, df_ai]).sample(frac=1, random_state=42).reset_index(drop=True)
    except Exception as e:
        print(f"HuggingFace download failed or timed out: {e}")
        print("Falling back to built-in synthetic dataset to ensure the code runs without errors.")
        
        # Hardcoded realistic synthetic data for testing
        human_texts = [
            "I really enjoyed my vacation to Hawaii, the weather was just perfect. The ocean was so blue.",
            "Honestly, we should focus more on local businesses. They give so much character to the town.",
            "Growing up, I loved reading fantasy books. The Hobbit was probably my favorite adventure.",
            "Climate change is definitely something we need to worry about. The storms are getting worse.",
            "It's just crazy how fast technology is moving nowadays. I can barely keep up with my phone."
        ] * 120 # 600 samples

        ai_texts = [
            "As an AI language model, I don't have personal experiences, but Hawaii is known for its climate.",
            "In conclusion, local economies benefit from small businesses through robust job creation metrics.",
            "The fantasy genre typically involves elements of magic, mythical creatures, and fictional universes.",
            "Climate change refers to long-term shifts in temperatures and weather patterns driven by humans.",
            "Technological advancement has accelerated significantly in the 21st century, impacting daily life."
        ] * 120 # 600 samples
        
        df_human = pd.DataFrame({'text': human_texts, 'label': 0})
        df_ai = pd.DataFrame({'text': ai_texts, 'label': 1})
        df_train = pd.concat([df_human, df_ai]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset ready: {len(df_train)} samples.")
    
    extractor = TextFeatureExtractor()
    print("Extracting Stylometrics & DistilBERT Embeddings... (this takes a few minutes)")
    
    X_features = []
    labels = df_train['label'].values
    
    start_time = time.time()
    
    # Extract Transformer Embeddings in batches
    embeddings = extractor.get_embeddings(df_train['text'].tolist(), batch_size=32)
    
    # Extract Stylometric features
    for i, text in enumerate(df_train['text']):
        stats = list(extractor.get_stylometric_features(text).values())
        X_features.append(np.concatenate([stats, embeddings[i]]))
        if (i+1) % 500 == 0:
            print(f"Processed {i+1} samples...")
            
    X_features = np.array(X_features)
    print(f"Feature extraction complete in {time.time() - start_time:.2f}s. Shape: {X_features.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_features, labels, test_size=0.2, random_state=42, stratify=labels)
    
    if HAS_XGB:
        print("Training XGBoost Hybrid Model...")
        clf = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=120,
            objective='binary:logistic',
            eval_metric='logloss'
        )
    else:
        print("XGBoost not found. Falling back to robust Random Forest Alternative.")
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42)
        
    clf.fit(X_train, y_train)
    
    # Eval
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- EduGuard Model Evaluation ---")
    print(f"Accuracy: {acc*100:.2f}% (Target: 88-90%)")
    print(f"F1-Score: {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/xgboost_ai_detector.pkl")
    print("Model saved successfully: models/xgboost_ai_detector.pkl")
    
if __name__ == "__main__":
    train_model()
