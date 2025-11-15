# backend/train_model.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load data (must exist)
train_path = "data/train.csv"
val_path = "data/val.csv"
if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise FileNotFoundError("Place train.csv and val.csv into data/ (see generate_synthetic.py)")

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)

# combine fields
train["text"] = train["merchant"].astype(str) + " " + train["description"].astype(str)
val["text"] = val["merchant"].astype(str) + " " + val["description"].astype(str)

# Vectorizer and model
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=20000)
X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, train["category"])

# Evaluate
y_pred = model.predict(X_val)
print("Classification report (validation):")
print(classification_report(val["category"], y_pred))

# Save
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("✅ Model and vectorizer saved to models/")
