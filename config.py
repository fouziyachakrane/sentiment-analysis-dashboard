"""
config.py — Configuration centralisée du projet
=================================================
Toutes les constantes (chemins, hyperparamètres, options d'entraînement)
sont définies ici pour éviter les "magic numbers" dispersés dans le code
et faciliter le déploiement (une seule source de vérité).
"""

import os

# ── Chemins de base ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

# ── Base de données ───────────────────────────────────────────────────────
DATABASE_PATH = os.path.join(DATABASE_DIR, "history.db")

# ── Fichiers de modèles sérialisés (Joblib) ─────────────────────────────
BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.joblib")
COMPARISON_RESULTS_PATH = os.path.join(SAVED_MODELS_DIR, "comparison_results.json")
MODEL_METADATA_PATH = os.path.join(SAVED_MODELS_DIR, "metadata.json")
EDA_DATA_PATH = os.path.join(SAVED_MODELS_DIR, "eda_data.json")
# Résultats de comparaison DistilBERT — générés à la main via
# notebooks/transformer_comparison.ipynb, à titre de référence uniquement
# (jamais entraîné/servi par l'app elle-même, voir services/prediction_service.py)
TRANSFORMER_REFERENCE_PATH = os.path.join(SAVED_MODELS_DIR, "transformer_reference.json")

# ── Dataset ──────────────────────────────────────────────────────────────
DATASET_NAME = "imdb"          # Dataset Hugging Face
SAMPLE_SIZE = None             # Nombre d'avis utilisés (None = dataset complet)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Vectorisation TF-IDF ─────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)

# ── Modèles comparés ─────────────────────────────────────────────────────
# Ordre = ordre d'affichage dans le tableau comparatif
MODEL_NAMES = [
    "Logistic Regression",
    "Naive Bayes",
    "Linear SVM",
    "Random Forest",
]

# ── API Flask ────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 5000))
DEBUG_MODE = False

# ── Explicabilité (LIME) ────────────────────────────────────────────────
LIME_NUM_FEATURES = 10          # Nombre de mots influents à afficher
LIME_NUM_SAMPLES = 500          # Échantillons de perturbation (perf vs précision)