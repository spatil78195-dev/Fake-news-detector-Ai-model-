# FakeGuard AI — Fake News Detector

AI-powered web application that detects whether a news article is **Fake** or **Real** using machine learning and NLP.

## Features

| Feature | Description |
|---------|-------------|
| ML prediction | TF-IDF + calibrated classifier with heuristic signals |
| URL checker | Extract and analyze articles from URLs |
| Image forensics | ELA + metadata analysis (optional deps) |
| Voice analysis | Transcribe audio and predict (optional deps) |
| Multi-language | Hindi/Marathi translation + prediction |
| Admin dashboard | Model status, retrain, news fetch |
| Self-learning | Scheduled news fetch + weekly retrain (MongoDB optional) |

## Quick start (local)

```bash
cd "Fake News Detector"
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python scripts/download_nltk.py
python train_model.py          # if model.pkl is missing
python app.py
```


## Deploy on Render

1. Push the repo to GitHub (include `model.pkl`, `vectorizer.pkl`, `model_meta.pkl`).
2. [Render](https://render.com) → **New Web Service** → connect repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt && python scripts/download_nltk.py`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Add environment variables from `.env.example` (`SECRET_KEY`, optional `MONGO_URI`, `NEWSAPI_KEY`).
5. Deploy.

Or use the included **`render.yaml`** blueprint.

## Deploy on Railway

1. Push to GitHub.
2. [Railway](https://railway.app) → **New Project** → **Deploy from GitHub**.
3. Railway reads **`railway.toml`** and **`Procfile`** automatically.
4. Set env vars in the Railway dashboard.
5. Generate a public domain under **Settings → Networking**.

**Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT`

## API

### `POST /predict`

```json
{ "text": "Your news article text..." }
```

Response:

```json
{
  "result": "Fake",
  "confidence": 94.72,
  "fake_prob": 94.72,
  "real_prob": 5.28,
  "word_count": 143
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

## Project structure

```
├── app.py                 Flask application
├── train_model.py         Train and save model
├── model.pkl              Trained classifier
├── vectorizer.pkl         TF-IDF vectorizer
├── model_meta.pkl         Threshold + metadata
├── requirements.txt       Python dependencies
├── Procfile               Render/Railway start command
├── core/                  ML pipeline, prediction, heuristics
├── features/              URL, image, voice, language modules
├── scheduler/             Auto fetch + retrain jobs
├── templates/             HTML pages
└── static/                CSS + JavaScript
```

## Environment variables

See **`.env.example`** for all options.

## Disclaimer

Educational use only. Always verify news from multiple trusted sources.
