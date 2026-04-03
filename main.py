from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from features import TextFeatureExtractor
from evaluator import ThinkingEvaluator

app = FastAPI(title="EduGuard API")

ML_MODEL = None
EXTRACTOR = None
EVALUATOR = ThinkingEvaluator()

def load_services():
    global ML_MODEL, EXTRACTOR
    if ML_MODEL is None:
        model_path = "models/xgboost_ai_detector.pkl"
        if not os.path.exists(model_path):
            raise Exception("Model not trained yet. Run train.py first to train the detector model.")
        ML_MODEL = joblib.load(model_path)
    if EXTRACTOR is None:
        EXTRACTOR = TextFeatureExtractor()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    try:
        load_services()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    text = request.text
    if len(text.strip()) < 20:
         raise HTTPException(status_code=400, detail="Text is too short for robust analysis. Please provide more context.")

    # ML Detection Pipeline
    stats = list(EXTRACTOR.get_stylometric_features(text).values())
    emb = EXTRACTOR.get_embeddings([text])[0]
    
    X = np.concatenate([stats, emb]).reshape(1, -1)
    
    # Predict Probability: index 1 is AI class
    ai_prob = float(ML_MODEL.predict_proba(X)[0][1])
    is_ai = ai_prob > 0.5
    
    # Critical Thinking Evaluator Pipeline
    eval_result = EVALUATOR.evaluate(text)
    
    return {
        "status": "success",
        "ai_probability": ai_prob,
        "is_ai_flagged": is_ai,
        "critical_thinking": eval_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
