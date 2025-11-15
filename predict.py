import joblib
import numpy as np

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_transaction(merchant, description):
    text = merchant + " " + description
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()
    print(f"Merchant: {merchant}\nDescription: {description}")
    print(f"Predicted Category: {pred} (Confidence: {prob:.2f})\n")



def explain_prediction(merchant, description):
    text = merchant + " " + description
    X = vectorizer.transform([text])
    pred_class = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = probs.max()
    
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[model.classes_.tolist().index(pred_class)]
    important_features = np.argsort(coefs)[-10:]
    keywords = [feature_names[i] for i in important_features]
    
    print(f"\nPrediction Explanation:")
    print(f"Category: {pred_class} (confidence={confidence:.2f})")
    print(f"Top keywords influencing decision: {keywords}")

predict_transaction("Starbucks #123 SF", "POS DEBIT coffee purchase")
predict_transaction("Amazon Mktp US", "online order")
predict_transaction("Shell 4421", "fuel transaction")
predict_transaction("Netflix", "monthly subscription")
predict_transaction("Apple Store", "in-app purchase")
