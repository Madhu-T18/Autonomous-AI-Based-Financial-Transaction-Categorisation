# app.py
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = "dev-key"  # change for production

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
TAXONOMY_PATH = "taxonomy.json"
FEEDBACK_PATH = "data/feedback_log.csv"

# Load taxonomy
with open(TAXONOMY_PATH, "r") as f:
    taxonomy = json.load(f)["categories"]

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    global model, vectorizer
    pred = None
    conf = None
    explanation = None
    merchant = ""
    description = ""

    if request.method == "POST":
        # handle batch upload separately
        if "batch_upload" in request.files and request.files["batch_upload"].filename != "":
            f = request.files["batch_upload"]
            df = pd.read_csv(f)
            if "merchant" not in df.columns or "description" not in df.columns:
                flash("CSV must contain 'merchant' and 'description' columns", "error")
                return redirect(url_for("index"))
            texts = df["merchant"].astype(str) + " " + df["description"].astype(str)
            model, vectorizer = load_model()
            if model is None:
                flash("Model not trained. Run backend/train_model.py first.", "error")
                return redirect(url_for("index"))
            X = vectorizer.transform(texts)
            df["predicted_category"] = model.predict(X)
            df["confidence"] = np.max(model.predict_proba(X), axis=1)
            out_path = "data/batch_results.csv"
            df.to_csv(out_path, index=False)
            return send_file(out_path, as_attachment=True)
        # single prediction flow
        merchant = request.form.get("merchant","").strip()
        description = request.form.get("description","").strip()
        if merchant == "" or description == "":
            flash("Enter merchant and description", "error")
            return redirect(url_for("index"))

        text = merchant + " " + description
        if model is None:
            flash("Model not trained. Run backend/train_model.py first.", "error")
            return redirect(url_for("index"))
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        conf = float(np.max(model.predict_proba(X)))

        # explainability: top positive coefficients for predicted class
        try:
            feat_names = vectorizer.get_feature_names_out()
            class_idx = list(model.classes_).index(pred)
            coefs = model.coef_[class_idx]
            top_inds = np.argsort(coefs)[-8:][::-1]
            explanation = [feat_names[i] for i in top_inds]
        except Exception as e:
            explanation = []

    return render_template("index.html",
                           categories=taxonomy,
                           predicted=pred,
                           confidence=conf,
                           explanation=explanation,
                           merchant=merchant,
                           description=description)

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    merchant = request.form.get("merchant","")
    description = request.form.get("description","")
    predicted = request.form.get("predicted","")
    correct = request.form.get("correct_category","")
    if correct == "":
        correct = predicted
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    row = {
        "merchant": merchant,
        "description": description,
        "predicted": predicted,
        "correct_category": correct,
        "timestamp": datetime.utcnow().isoformat()
    }
    df = pd.DataFrame([row])
    df.to_csv(FEEDBACK_PATH, mode="a", index=False, header=not os.path.exists(FEEDBACK_PATH))
    flash("Feedback saved. Thank you!", "success")
    return redirect(url_for("index"))

@app.route("/retrain", methods=["POST"])
def retrain():
    # Admin endpoint: call retrain script
    from backend import retrain_model as retrain_module  # uses file backend/retrain_model.py
    try:
        # execute retrain logic by running that file's main flow (simple import executes)
        # We will call the function by executing as script; safer approach: call subprocess
        import subprocess, sys
        proc = subprocess.run([sys.executable, "backend/retrain_model.py"], capture_output=True, text=True)
        if proc.returncode != 0:
            flash(f"Retrain failed: {proc.stderr}", "error")
        else:
            # reload model
            global model, vectorizer
            model, vectorizer = load_model()
            flash("Model retrained and reloaded.", "success")
    except Exception as e:
        flash(f"Retrain error: {e}", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
