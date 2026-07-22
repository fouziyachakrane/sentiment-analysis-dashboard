# 🎬 Sentiment Analysis Pro — IMDB Reviews

Full-stack sentiment analysis application on movie reviews (**IMDB dataset, 50,000 reviews** — combined HuggingFace train+test splits), turned from an academic mini-project (Master's in Data Science and Analytics, FP Safi — Cadi Ayyad University) into a complete product: model comparison, interactive dashboard, explainability (LIME), persistent history and REST API.

## ✨ Features

- **4 models compared automatically**: Logistic Regression, Naive Bayes, Linear SVM, Random Forest — the best one (F1-score) is selected and served in production
- **Optional DistilBERT reference row**: results from a separate Transformer fine-tuning notebook can be displayed read-only in the Models tab, without ever being served in production (see "Comparison with a Transformer model" below)
- **Interactive dashboard**: class distribution, model performance, most frequent words (Chart.js)
- **Real-time prediction** with confidence score
- **Explainability (LIME)**: words that most influenced each decision, with signed weight
- **Persistent history** (SQLite): every prediction is logged (text, result, confidence, date)
- **REST API** (`/api/predict`) reusable independently of the frontend
- **Pre-trained, serialized model** (Joblib): instant loading, no retraining on startup

## 🏗️ Architecture

```
sentiment-analysis-pro/
├── app.py                      # Flask entry point (API + frontend)
├── config.py                   # Centralized configuration
├── models/
│   └── train_models.py         # Training + comparison of the 4 models
├── services/
│   ├── prediction_service.py   # Prediction/confidence/history orchestration
│   ├── explainability.py       # LIME explainability
│   └── eda_service.py          # Merges dataset data + model metrics
├── database/
│   └── history_db.py           # Prediction history (SQLite)
├── utils/
│   └── text_preprocessing.py   # Shared NLP cleaning pipeline
├── static/
│   └── index.html              # Frontend (Chart.js, light theme)
├── notebooks/
│   └── transformer_comparison.ipynb  # DistilBERT fine-tuning vs TF-IDF+LogReg baseline
├── saved_models/                # best_model.joblib, comparison, metadata, EDA, transformer_reference.json (generated)
├── data/                        # Optional local data
├── requirements.txt
├── Procfile                     # Render / Heroku-style deployment
└── Dockerfile                   # Containerized deployment (HF Spaces, Render)
```

**Key principle**: `models/train_models.py` trains and serializes the model once; `app.py` only **loads** it (Joblib) — no more hot retraining on every startup like in the initial academic version.

## 🚀 Installation

```bash
git clone <your-repo>
cd sentiment-analysis-pro
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🧠 Model training

```bash
python -m models.train_models
```

Downloads IMDB (Hugging Face `datasets`), cleans the text, trains the 4 models, prints the comparison table to the console, and saves to `saved_models/`:
- `best_model.joblib` — full pipeline (TF-IDF + best classifier)
- `comparison_results.json` — metrics for the 4 models
- `metadata.json` — selected model, detailed metrics, confusion matrix
- `eda_data.json` — exploratory dataset statistics

## ▶️ Run

```bash
python app.py
```

App available at **http://localhost:5000**.

## 📡 REST API

| Method | Route                  | Description                                  |
|--------|-------------------------|-----------------------------------------------|
| POST   | `/api/predict`          | `{"text": "Amazing movie"}` → `{"prediction": "Positive", "confidence": 98.2}` |
| POST   | `/api/explain`          | Same + most influential words (LIME)          |
| GET    | `/api/eda`               | Dataset statistics + model metrics             |
| GET    | `/api/stats`             | User prediction statistics                     |
| GET    | `/api/history?limit=50`  | Prediction history                             |
| GET    | `/api/compare-models`    | Comparison table of the 4 models (+ optional DistilBERT reference row) |

## 🛠️ Tech stack

**Backend**: Python, Flask, Scikit-Learn, Joblib, SQLite, LIME, NLTK
**Frontend**: HTML/CSS/JS, Chart.js
**Data**: IMDB Dataset (Hugging Face `datasets`), TF-IDF (uni+bigrams)

## 📊 Results

The best model is automatically selected on F1-score; exact metrics (accuracy, precision, recall, F1, training time) are available in `saved_models/comparison_results.json` after training, and displayed in the **Models** tab of the dashboard.

## ☁️ Deployment

**Render**: connect the repo, `Procfile` already configured (`gunicorn app:app`). Remember to run `python -m models.train_models` as a build step (or commit `saved_models/` if size allows).

**Hugging Face Spaces (Docker SDK)**: push the provided `Dockerfile` — it trains the model at build time and serves the app on port `7860`.

**Streamlit Cloud**: requires porting the frontend to Streamlit (not included here; the Flask API + `services/` can be reused as-is as the business logic layer).

## 🔬 Comparison with a Transformer model

`notebooks/transformer_comparison.ipynb` fine-tunes `distilbert-base-uncased` on the same dataset (same 80/20 split and test set) and compares accuracy, F1-score, training time and inference time with the TF-IDF + Logistic Regression baseline used in production. Goal: document the performance/cost trade-off between classic ML and Deep Learning, not to replace the production pipeline.

## 🔭 Roadmap

- Adaptation to Moroccan dialect (Darija)
- Authentication for per-user history

## 👥 Author

Fouziya CHAKRANE
Master's in Data Science and Analytics — Polydisciplinary Faculty of Safi, Cadi Ayyad University (2025/2026)
