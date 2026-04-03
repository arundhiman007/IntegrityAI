# 🛡️ EduGuard: AI-Powered Academic Integrity & Critical Thinking System

EduGuard is an intelligent educational tool that addresses the increasing prevalence of Generative AI in academic submissions. It uses a **Hybrid ML Architecture** to detect AI-generated content with high precision while simultaneously evaluating the student's depth of learning, logical reasoning, and originality.

## 🌟 Key Features
- **Authenticity Analysis**: Detects AI-generated essays using `distilbert` semantic embeddings combined with offline stylometric analysis (perplexity, syntax variation, vocabulary richness).
- **Critical Thinking Evaluator**: Utilizes Large Language Models (LLMs) instructed with strict academic rubrics to grade the conceptual quality of the submission.
- **Standalone Dashboard**: A beautiful, modern Streamlit UI optimized with glassmorphic CSS.

## 🛠️ Project Architecture
1. **`features.py`**: The mathematical core. Extracts text statistics and loads HuggingFace configurations.
2. **`train.py`**: Downloads the DAIGT v2 dataset, balances the labels, extracts features, and trains the `XGBoost` model.
3. **`evaluator.py`**: The agentic Langchain wrapper connecting to OpenAI's GPT for conceptual grading.
4. **`main.py`**: A FastAPI backend for exposing the system as REST endpoints.
5. **`app.py`**: The interactive Streamlit user testing platform.

## 🚀 Setup & Installation
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables:
   Rename `.env.example` to `.env` and add your OpenAI Key for the Advanced critical thinking capabilities.
   *(Note: The system smartly falls back to mock logic evaluations if you are working offline or don't have a key!)*

3. Train the model:
   ```bash
   python train.py
   ```

4. Run the Streamlit User Interface:
   ```bash
   streamlit run app.py
   ```
