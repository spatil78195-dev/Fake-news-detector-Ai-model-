# Deploy Fake News Detector on Vercel

## What is included in the serverless bundle

| Included | Excluded (`.vercelignore`) |
|----------|----------------------------|
| `model.pkl`, `vectorizer.pkl`, `model_meta.pkl` | `dataset/` CSVs (~1.7 MB) |
| `nltk_data/` bundled stopwords | `venv/`, logs, training scripts |
| Core prediction + heuristics | `opencv`, `newspaper3k`, `pydub`, scheduler |
| URL check (requests + BeautifulSoup) | Weekly retrain / news fetch jobs |

**Estimated install size:** ~120–180 MB (under Vercel’s 250 MB serverless limit).

## Prerequisites

1. [Vercel account](https://vercel.com)
2. [Vercel CLI](https://vercel.com/docs/cli): `npm i -g vercel`
3. Trained model files committed at project root:
   - `model.pkl`
   - `vectorizer.pkl`
   - `model_meta.pkl`

If missing, run locally:

```bash
pip install -r requirements-full.txt
python generate_dataset.py   # if needed
python train_model.py
```

## Deploy

```bash
cd "Fake News Detector"
vercel login
vercel          # first deploy (preview)
vercel --prod   # production
```

Or connect the GitHub repo in the Vercel dashboard (Framework: **Other**).

## Environment variables (Vercel dashboard)

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | Recommended | Flask session secret |
| `ADMIN_PASSWORD` | Optional | Admin retrain (disabled on Vercel anyway) |
| `MONGODB_URI` | Optional | Not used unless you add full `requirements-full.txt` |

## Features on Vercel

| Feature | Status |
|---------|--------|
| Text prediction (`/predict`) | Yes |
| Sample articles, health, PDF UI | Yes |
| URL checker | Yes (BeautifulSoup; no newspaper3k) |
| Image / voice forensics | Disabled (heavy deps excluded) |
| Multi-language | Detect only; translation needs `deep-translator` locally |
| Admin retrain / scheduler | Disabled |

## Local development (full features)

```bash
pip install -r requirements-full.txt
python app.py
```

## Troubleshooting

- **Model not loaded:** Ensure `model.pkl` and `vectorizer.pkl` are in the repo and not listed in `.vercelignore`.
- **Build too large:** Do not add `requirements-full.txt` as the install file; Vercel must use root `requirements.txt`.
- **Cold start slow:** First request after idle may take 5–15 s while sklearn loads.
