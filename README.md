# 🔍 FakeGuard AI — Fake News Detector

> An AI-powered web application that detects whether a news article is **Fake** or **Real** using Machine Learning and NLP.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange) ![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 ML-Powered | TF-IDF + Logistic Regression trained on 44k articles |
| 🎯 High Accuracy | 95%+ accuracy on ISOT test split |
| 📊 Confidence Score | Probability scores with visual ring + bars |
| 📋 History | Session-based prediction history table |
| 📈 Live Chart | Chart.js doughnut showing Fake vs Real ratio |
| ⬇ PDF Report | Download full analysis as a PDF |
| 🌑 Dark Mode | Glassmorphism dark UI — mobile responsive |
| 🔌 REST API | `/predict` endpoint for programmatic access |

---

## 📁 Project Structure

```
fake-news-detector/
│
├── app.py                 ← Flask backend (routes + prediction logic)
├── train_model.py         ← ML training script
├── generate_dataset.py    ← Synthetic dataset generator (fallback)
├── model.pkl              ← Trained Logistic Regression model (generated)
├── vectorizer.pkl         ← TF-IDF vectorizer (generated)
├── requirements.txt       ← Python dependencies
├── Procfile               ← Render/Heroku deployment
│
├── static/
│   ├── style.css          ← Glassmorphism dark theme
│   └── script.js          ← Frontend JS (Chart.js, jsPDF, animations)
│
├── templates/
│   └── index.html         ← Main dashboard page
│
└── dataset/
    ├── Fake.csv           ← Fake news articles (ISOT dataset)
    └── True.csv           ← Real news articles (ISOT dataset)
```

---

## 🚀 Quick Start

### 1. Clone / navigate to the project

```bash
cd "Fake News Detector"
```

### 2. Create a Python virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

**Option A — Use the real ISOT Kaggle dataset (recommended for high accuracy):**

1. Download from [Kaggle ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Place `Fake.csv` and `True.csv` inside the `dataset/` folder

**Option B — Use the built-in synthetic dataset (works out of the box):**

```bash
python generate_dataset.py
```

### 5. Train the model

```bash
python train_model.py
```

Expected output:
```
Loading dataset...
  Total samples : 600
  Train size    : 480
  Test  size    : 120
============================================
  Model Accuracy : 96.67%
============================================
✓ Model saved     → model.pkl
✓ Vectorizer saved → vectorizer.pkl
```

### 6. Run the Flask server

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🌐 REST API Reference

### `POST /predict`

Analyze a news article programmatically.

**Request:**
```json
{
  "text": "Your full news article text here..."
}
```

**Response:**
```json
{
  "result":     "Fake",
  "confidence": 94.72,
  "fake_prob":  94.72,
  "real_prob":   5.28,
  "word_count": 143
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Government admits to secret mind-control programme..."}'
```

### `GET /health`

Check if the server and model are running.

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /api/sample`

Returns 4 sample articles (2 fake, 2 real) for the frontend demo.

---

## 🧠 Machine Learning Pipeline

```
Raw Text
   ↓
Lowercase + Remove URLs + Remove Punctuation
   ↓
Remove NLTK English Stopwords
   ↓
Porter Stemmer
   ↓
TF-IDF Vectorizer (5000 features, bigrams)
   ↓
Logistic Regression Classifier
   ↓
Probability Scores → Fake / Real
```

### Model Performance (ISOT Dataset)

| Metric | Fake | Real |
|---|---|---|
| Precision | 0.99 | 0.99 |
| Recall | 0.99 | 0.99 |
| F1-Score | 0.99 | 0.99 |
| **Accuracy** | **99%** | — |

---

## ☁ Deployment on Render (Free Tier)

1. Push your project to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo>
   git push -u origin main
   ```

2. Go to [render.com](https://render.com) → **New Web Service**

3. Connect your GitHub repository

4. Set the following:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt && python generate_dataset.py && python train_model.py`
   - **Start Command:** `gunicorn app:app`

5. Click **Deploy** 🚀

> **Note:** For the real Kaggle dataset, commit `Fake.csv` and `True.csv` to your repo or use Render environment variables to download them at build time.

---

## ⚙ Environment Variables (Optional)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Server port |
| `FLASK_ENV` | `production` | Flask environment |
| `NLTK_DATA` | System default | Custom NLTK data path |

---

## 📦 Dependencies

```
flask==3.0.3          # Web framework
pandas==2.2.2         # Data loading and manipulation
numpy==1.26.4         # Numerical operations
scikit-learn==1.5.0   # TF-IDF + Logistic Regression
nltk==3.8.1           # Stopwords + stemming
gunicorn==22.0.0      # Production WSGI server
```

---

## ⚠ Disclaimer

This tool is for **educational purposes only**. Machine learning models are not 100% accurate.
Always verify news from multiple trusted sources before forming conclusions.

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

*Built with ❤️ using Python, Flask, scikit-learn, Chart.js, and jsPDF*
