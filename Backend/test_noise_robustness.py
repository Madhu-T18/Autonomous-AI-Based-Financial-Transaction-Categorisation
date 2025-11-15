import joblib
import pandas as pd
import random

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

samples = [
    "Starbcks coffe purchse",
    "Ubre ride to airport",
    "Amzon order delivery",
    "Neflix monthly subscriptn",
    "Shell gas refill",
    "KFC meal order",
]

texts = pd.Series(samples)
X = vectorizer.transform(texts)
preds = model.predict(X)
probs = model.predict_proba(X).max(axis=1)

for t, p, c in zip(samples, preds, probs):
    print(f"{t:35s} → {p:12s} ({c:.2f})")
