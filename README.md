# FakeGuard AI — Fake News Detector

AI-powered web app that classifies news articles as **Fake** or **Real** using TF-IDF + machine learning, with optional URL, image, and voice checks (full features locally).

## Quick start (local)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements-full.txt
python train_model.py          # if model.pkl missing
python app.py
```

Open **http://localhost:5000**

## Deploy on Vercel (serverless)

Optimized bundle under Vercel’s size limits:

- **Slim deps:** `requirements.txt` (no OpenCV, newspaper3k, pandas, scheduler)
- **Excluded:** `dataset/`, `venv/`, training scripts (see `.vercelignore`)
- **Included:** pre-trained `model.pkl`, `vectorizer.pkl`, bundled `nltk_data/`

```bash
vercel --prod
```

Full guide: **[DEPLOY_VERCEL.md](DEPLOY_VERCEL.md)**

| On Vercel | Local (`requirements-full.txt`) |
|-----------|----------------------------------|
| Text prediction | All of the left |
| URL check (BeautifulSoup) | + newspaper3k |
| — | Image / voice / admin retrain |

## API

`POST /predict` with JSON `{ "text": "..." }` → `{ "result", "confidence", "fake_prob", "real_prob" }`

## Disclaimer

Educational use only. Always verify news from multiple trusted sources.
