# Autonomous-AI-Based-Financial-Transaction-Categorisation

**Offline AI system to automatically categorize transaction descriptions (e.g., "Uber Trip" → Transport) with explainability, feedback, and retraining — no external APIs required.**

---

## 📸 Demo
<img width="1722" height="951" alt="Screenshot (752)" src="https://github.com/user-attachments/assets/fe9c1a05-8e4b-4d99-850e-d43139b1aa66" />


## 🚀 Quick Start

```bash
git clone https://github.com/Madhu-T18/Autonomous-AI-Based-Financial-Transaction-Categorisation.git
cd FinSort-AI

pip install -r requirements.txt

# Train model
python train_model.py

# Run app
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 🧪 Example Workflow

1. Enter transaction:

   ```
   Starbucks Coffee
   purchase coffee
   ```

2. Click **Predict**

3. Output:

   * Category: `Food`
   * Confidence: `0.98`
   * Top keywords: `coffee`, `cafe`, etc.

4. If incorrect:

   * Enter correct category
   * Submit feedback

5. Click **Retrain Model** to improve future predictions

---

## 🚀 Features

* Real-time prediction via web UI
* Batch CSV upload for bulk classification
* Confidence score for each prediction
* Explainable output (top contributing n-grams)
* Feedback system to correct predictions
* One-click model retraining
* Fully offline (no APIs, no external calls)

---

## 📄 CSV Format (Batch Upload)

```csv
merchant,description
Starbucks Coffee,purchase coffee
Uber,ride to airport
Amazon,online order
```

Output will include:

* predicted_category
* confidence score

---

## 🧠 How It Works

* Text preprocessing (cleaning, normalization)
* TF-IDF feature extraction:

  * Word n-grams (1–2)
  * Character n-grams (3–5)
* Logistic Regression classifier
* Outputs:

  * Predicted category
  * Confidence score
  * Top contributing keywords

---

## 📊 Model Performance

* Macro F1 Score: **> 0.90**
* Evaluated on mixed real + synthetic dataset

> Note: Accuracy may vary depending on dataset quality and category distribution.

---

## ⚙️ Configuration

### Edit Categories

Modify:

```
config/taxonomy.json
```

* Add / remove / rename categories
* No code changes required

---

## 🔁 Retraining

To retrain the model:

```bash
python train_model.py
```

Training uses:

* `data/train.csv`
* optional feedback data (if integrated)

Outputs:

* `model.pkl`
* `vectorizer.pkl`

---

## 📂 Project Structure

```
FinSort-AI/
│── app.py
│── train_model.py
│── model.pkl
│── vectorizer.pkl
│── config/
│   └── taxonomy.json
│── data/
│   ├── train.csv
│   ├── val.csv
│   └── synthetic_gen.py
│── explainability/
│   └── top_tokens.json
│── templates/
│   └── index.html
│── static/
│   └── style.css
└── README.md
```

---

## ⚠️ Limitations

* Explainability currently shows raw n-grams (not fully user-friendly)
* Performance depends on training data quality
* Single-label classification only

---

## 📦 Use Cases

* Expense tracking apps
* Bank transaction labeling
* Accounting automation
* Financial analytics dashboards

---

## 📚 Future Improvements

* Transformer-based models (DistilBERT / FinBERT)
* Multi-label classification
* Improved explainability (human-readable keywords)
* Real-time streaming pipeline
* Mobile deployment (TensorFlow Lite)

---

## 📄 License

MIT License
