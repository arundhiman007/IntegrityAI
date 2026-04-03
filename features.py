import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
import torch

class TextFeatureExtractor:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.use_mock = False
        except Exception as e:
            print(f"Warning: Could not connect to HuggingFace or load DistilBERT ({e}). Using mock embeddings for offline functionality.")
            self.use_mock = True

    def get_stylometric_features(self, text):
        """Extract lightweight, non-library dependent stylometric features."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if len(sentences) == 0:
            sentences = [text]

        words = re.findall(r'\b\w+\b', text.lower())
        num_words = len(words)

        # 1. Average sentence length
        word_counts = [len(re.findall(r'\b\w+\b', s.lower())) for s in sentences]
        avg_sentence_len = np.mean(word_counts) if word_counts else 0

        # 2. Sentence length variation
        std_sentence_len = np.std(word_counts) if word_counts else 0

        # 3. Vocabulary richness (Type-Token Ratio)
        unique_words = set(words)
        ttr = len(unique_words) / num_words if num_words > 0 else 0

        # 4. Long word frequency (>6 chars)
        long_words = [w for w in words if len(w) > 6]
        long_word_freq = len(long_words) / num_words if num_words > 0 else 0

        return {
            "avg_sentence_len": avg_sentence_len,
            "std_sentence_len": std_sentence_len,
            "ttr": ttr,
            "long_word_freq": long_word_freq
        }

    def get_embeddings(self, texts, batch_size=32):
        """Extract [CLS] token embedding using batching, gracefully falling back to offline mode."""
        if getattr(self, 'use_mock', False):
            # Return dummy structural embeddings (768 dimensions representing distilbert shape)
            np.random.seed(42)
            return np.random.rand(len(texts), 768)

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_text = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                # Take the [CLS] token representation
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.extend(cls_embeddings)
        return np.array(all_embeddings)
