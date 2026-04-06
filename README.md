# 🚀 Autonomous AI-Based Financial Transaction Categorisation

**Offline AI system to automatically categorize transaction descriptions (e.g., "Uber Trip" → Transport) with explainability, feedback, and retraining — no external APIs required.**

---

## 🌐 Live Demo

Deployed on AWS EC2:

```
http://<3.228.18.6>
```

---

## 📸 Demo

---
<img width="1920" height="1080" alt="Screenshot (777)" src="https://github.com/user-attachments/assets/a86e0a45-d82f-4dc9-91e5-02baa2c77d9f" />
<img width="1920" height="906" alt="Screenshot (782)" src="https://github.com/user-attachments/assets/e3da9b32-13ca-4eb0-b502-9fef2288ff29" />
<img width="1920" height="903" alt="Screenshot (780)" src="https://github.com/user-attachments/assets/13f82f05-991e-4479-8fab-ed3ca149800c" />
<img width="1920" height="977" alt="Screenshot (784)" src="https://github.com/user-attachments/assets/a91ebbcb-ece7-4ea3-b312-31d163eadac1" />




## 🚀 Quick Start (Local Setup)

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

## 🐳 Docker Setup (Containerization)

### Build Docker Image

```bash
docker build -t myapp .
```

### Tag Image

```bash
docker tag myapp 230038/autonomous-ai-based-financial-transaction-categorisation
```

### Push to Docker Hub

```bash
docker push 230038/autonomous-ai-based-financial-transaction-categorisation
```

---

## ☁️ Cloud Deployment (AWS EC2)

Deployed using **Amazon EC2**

### Steps:

1. Launch EC2 (Ubuntu)
2. Install Docker
3. Pull image:

```bash
docker pull 230038/autonomous-ai-based-financial-transaction-categorisation
```

4. Run container:

```bash
docker run -d -p 80:5000 230038/autonomous-ai-based-financial-transaction-categorisation
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

5. Click **Retrain Model** to improve predictions

---

## 🚀 Features

* Real-time prediction via web UI
* Batch CSV upload for bulk classification
* Confidence score for each prediction
* Explainable output (top contributing n-grams)
* Feedback system for corrections
* One-click model retraining
* Fully offline (no APIs)

---

## 📄 CSV Format (Batch Upload)

```csv
merchant,description
Starbucks Coffee,purchase coffee
Uber,ride to airport
Amazon,online order
```

---

## 🧠 How It Works

* Text preprocessing (cleaning, normalization)
* TF-IDF feature extraction:

  * Word n-grams (1–2)
  * Character n-grams (3–5)
* Logistic Regression classifier

Outputs:

* Predicted category
* Confidence score
* Top contributing keywords

---

## 📊 Model Performance

* Macro F1 Score: **> 0.90**
* Evaluated on mixed real + synthetic dataset

---

## ⚙️ Configuration

Edit categories:

```
config/taxonomy.json
```

---

## 🔁 Retraining

```bash
python train_model.py
```

Outputs:

* model.pkl
* vectorizer.pkl

---

## 📂 Project Structure

```
FinSort-AI/
│── app.py
│── train_model.py
│── model.pkl
│── vectorizer.pkl
│── config/
│── data/
│── templates/
│── static/
└── README.md
```

---

## ⚠️ Limitations

* Explainability uses raw n-grams
* Depends on training data quality
* Single-label classification

---

## 📦 Use Cases

* Expense tracking apps
* Bank transaction labeling
* Accounting automation
* Financial dashboards

---

## 📚 Future Improvements

* Transformer models (BERT / FinBERT)
* Multi-label classification
* Better explainability
* CI/CD using Jenkins
* Domain + auto-scaling deployment

---

## 👨‍💻 Author

Madhu T

---

## 📄 License

MIT License
