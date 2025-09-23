from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import joblib
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = FastAPI(title="NLP API")

# Config
MODEL_DIR = "models"
PKL_PATH = os.path.join(MODEL_DIR, "hazard_svm_pipeline.pkl")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_LEN = 64

# Load pipeline
pipeline = joblib.load(PKL_PATH)
clf = pipeline["clf"]
scaler = pipeline["scaler"]
pca = pipeline["pca"]
le = pipeline["label_encoder"]

# Load tokenizer and BERT
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_DIR)
bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)

# Clean text function
def clean_text(s: str):
    return s.strip().lower()

# Get BERT embeddings
def get_bert_embeddings(texts: list):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = bert(**enc)
        embeddings.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)

# Predict function
def predict_texts(texts: list, threshold: float = 0.6):
    cleaned = [clean_text(t) for t in texts]
    emb = get_bert_embeddings(cleaned)
    emb_scaled = scaler.transform(emb)
    emb_pca = pca.transform(emb_scaled)
    probs = clf.predict_proba(emb_pca)
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    results = []
    for text, pred, conf in zip(texts, preds, confidences):
        label = le.inverse_transform([pred])[0] if conf >= threshold else "uncertain"
        results.append({"text": text, "prediction": label, "confidence": float(conf)})
    return results

# Request model
class RedditPredictRequest(BaseModel):
    reddit_query: str
    # limit: Optional[int] = 25

@app.post("/reddit_predict")
def reddit_predict(req: RedditPredictRequest):
    url = f"https://www.reddit.com/search.json?q={req.reddit_query}"
    headers = {"User-Agent": "nlp-api/0.1"}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        reddit_data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reddit API error: {e}")

    posts = []
    texts_to_predict = []

    for child in reddit_data.get("data", {}).get("children", []):
        d = child["data"]
        title = d.get("title", "")
        selftext = d.get("selftext") or ""
        media = d.get("url_overridden_by_dest") or d.get("url") or ""
        posts.append({"title": title, "selftext": selftext, "media": media})
        texts_to_predict.append(title + " " + selftext)

    predictions = predict_texts(texts_to_predict)

    for post, pred in zip(posts, predictions):
        post["prediction"] = pred["prediction"]
        post["confidence"] = pred["confidence"]

    return {"posts": posts}
