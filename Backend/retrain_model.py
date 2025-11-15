# backend/retrain_model.py
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

os.makedirs("models", exist_ok=True)
train_path = "data/train.csv"
feedback_path = "data/feedback_log.csv"

if not os.path.exists(train_path):
    raise FileNotFoundError("data/train.csv not found")

train = pd.read_csv(train_path)
train["text"] = train["merchant"].astype(str) + " " + train["description"].astype(str)

if os.path.exists(feedback_path):
    fb = pd.read_csv(feedback_path)
    # feedback expected columns: merchant, description, correct_category
    if "correct_category" not in fb.columns:
        raise ValueError("feedback_log.csv must have column 'correct_category'")
    fb = fb.rename(columns={"correct_category": "category"})
    fb["text"] = fb["merchant"].astype(str) + " " + fb["description"].astype(str)
    combined = pd.concat([train[["text","category"]], fb[["text","category"]]], ignore_index=True)
    print(f"Retraining on {len(combined)} rows (including feedback).")
else:
    combined = train[["text","category"]]
    print(f"No feedback found. Retraining on {len(combined)} rows (original train only).")

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=20000)
X = vectorizer.fit_transform(combined["text"])
y = combined["category"]

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X, y)

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("✅ Retrained model saved to models/")
