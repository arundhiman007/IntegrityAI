import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
from features import TextFeatureExtractor

def evaluate_model():
    model_path = "models/xgboost_ai_detector.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run train.py first.")
        return
        
    print("Loading Trained XGBoost Model...")
    clf = joblib.load(model_path)
    
    print("Fetching Evaluation Dataset (using a distinct subset from DAIGT V2)...")
    try:
        print("Attempting to load evaluation dataset from HuggingFace...")
        from datasets import load_dataset
        # We sample an entirely different unseen slice of the dataset using a different random seed
        dataset = load_dataset("thedrcat/daigt-v2-train-dataset", split="train", trust_remote_code=True)
        df = dataset.to_pandas()
        df = df[['text', 'label']].dropna()
        
        df_human = df[df['label'] == 0].sample(int(min(200, len(df[df['label']==0]))), random_state=99)
        df_ai = df[df['label'] == 1].sample(int(min(200, len(df[df['label']==1]))), random_state=99)
        df_eval = pd.concat([df_human, df_ai]).sample(frac=1, random_state=99).reset_index(drop=True)
        
    except Exception as e:
        print(f"HuggingFace dataset error: {e}")
        print("Falling back to built-in evaluation synthetic dataset.")
        human_eval = [
            "We had a great time at the park yesterday. The kids loved the slides and the ice cream.",
            "I'm not really sure if the new policy makes sense for everyone in the company, to be honest.",
            "Can you believe how expensive groceries have gotten lately? It's ridiculous."
        ] * 40 # 120 samples
        
        ai_eval = [
            "The implementation of the new corporate policy has generated diverse perspectives among the staff.",
            "Rising inflation rates have directly contributed to the increased cost of essential consumer goods.",
            "Parks serve as vital community spaces that facilitate recreational activities and social cohesion."
        ] * 40 # 120 samples
        
        df_human = pd.DataFrame({'text': human_eval, 'label': 0})
        df_ai = pd.DataFrame({'text': ai_eval, 'label': 1})
        df_eval = pd.concat([df_human, df_ai]).sample(frac=1, random_state=99).reset_index(drop=True)

    print(f"Evaluation Dataset Ready: {len(df_eval)} samples loaded.")
    
    extractor = TextFeatureExtractor()
    print("Extracting features for the evaluation set... (This takes a moment)")
    
    X_features = []
    y_true = df_eval['label'].values
    
    start_time = time.time()
    embeddings = extractor.get_embeddings(df_eval['text'].tolist(), batch_size=32)
    
    for i, text in enumerate(df_eval['text']):
        stats = list(extractor.get_stylometric_features(text).values())
        X_features.append(np.concatenate([stats, embeddings[i]]))
        
    X_features = np.array(X_features)
    print(f"Feature Extraction completed in {time.time() - start_time:.2f} seconds.")
    
    print("\nPredicting on Evaluation Set...")
    y_pred = clf.predict(X_features)
    
    # Calculate robust metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("        EVALUATION ACCURACY REPORT      ")
    print("="*40)
    print(f"Overall Accuracy : {acc * 100:.2f}%")
    print(f"Overall F1-Score : {f1:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["Human (0)", "AI (1)"]))
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(f"True Human (TN): {cm[0][0]}  |  False AI (FP): {cm[0][1]}")
    print(f"False Human (FN): {cm[1][0]} |  True AI (TP): {cm[1][1]}")
    print("========================================")

if __name__ == "__main__":
    evaluate_model()
