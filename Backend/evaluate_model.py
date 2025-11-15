import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load model and vectorizer ---
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# --- Load validation dataset ---
val = pd.read_csv("data/val.csv")
val["text"] = val["merchant"].astype(str) + " " + val["description"].astype(str)

# --- Transform and predict ---
X_val = vectorizer.transform(val["text"])
y_true = val["category"]
y_pred = model.predict(X_val)

# --- Evaluation metrics ---
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=3))

# --- Confusion Matrix Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', 
            xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
