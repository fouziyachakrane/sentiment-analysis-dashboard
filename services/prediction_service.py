"""
services/prediction_service.py — Orchestration de la prédiction
===================================================================
Couche métier entre l'API Flask et les briques bas niveau : charge le
meilleur modèle (une seule fois, mis en cache), nettoie le texte,
prédit le sentiment avec un score de confiance, journalise en base
SQLite et, à la demande, fournit l'explication LIME.
"""

import json
from functools import lru_cache

import joblib

import config
from utils.text_preprocessing import clean_text
from services.explainability import explain_prediction, get_predict_proba_fn, CLASS_NAMES
from database import history_db


@lru_cache(maxsize=1)
def load_model():
    """
    Charge le pipeline (TF-IDF + meilleur classifieur) sauvegardé par
    models/train_models.py. Mis en cache : le modèle n'est désérialisé
    qu'une seule fois, au premier appel (plus de ré-entraînement au
    démarrage de l'application).
    """
    return joblib.load(config.BEST_MODEL_PATH)


def predict_sentiment(text: str, with_explanation: bool = False, save_to_history: bool = True) -> dict:
    """
    Prédit le sentiment d'un avis brut.

    Args:
        text: avis client brut (non nettoyé)
        with_explanation: si True, ajoute les mots les plus influents (LIME)
        save_to_history: si True, journalise la prédiction en base SQLite

    Returns:
        {
            "original_text": str, "cleaned_text": str,
            "prediction": "positive" | "negative", "confidence": float,
            "explanation": [...]  # présent seulement si with_explanation=True
        }

    Raises:
        ValueError: texte vide ou vide après nettoyage (ex: uniquement
        de la ponctuation/stop words).
    """
    if not text or not text.strip():
        raise ValueError("Le texte fourni est vide.")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Le texte ne contient aucun mot exploitable après nettoyage.")

    pipeline = load_model()
    proba_fn = get_predict_proba_fn(pipeline)
    probabilities = proba_fn([cleaned])[0]  # [proba_negative, proba_positive]

    predicted_idx = int(probabilities.argmax())
    prediction = CLASS_NAMES[predicted_idx]
    confidence = round(float(probabilities[predicted_idx]) * 100, 2)

    result = {
        "original_text": text,
        "cleaned_text": cleaned,
        "prediction": prediction,
        "confidence": confidence,
    }

    if save_to_history:
        history_db.save_prediction(text, cleaned, prediction, confidence)

    if with_explanation:
        result["explanation"] = explain_prediction(pipeline, cleaned)

    return result


def get_model_metadata() -> dict:
    """Métadonnées du modèle actuellement chargé (nom, métriques, date d'entraînement)."""
    with open(config.MODEL_METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_comparison_results() -> list[dict]:
    """Tableau comparatif des 4 modèles entraînés (pour l'onglet dashboard).

    Ajoute en fin de liste, si disponible, une ligne de référence
    DistilBERT (voir notebooks/transformer_comparison.ipynb) : ce modèle
    n'est ni entraîné ni servi par l'application, il est marqué
    `reference_only: true` et le frontend l'affiche différemment (pas de
    concurrence possible avec le "meilleur modèle" réellement déployé).
    """
    with open(config.COMPARISON_RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    try:
        with open(config.TRANSFORMER_REFERENCE_PATH, encoding="utf-8") as f:
            transformer_ref = json.load(f)
        # On n'affiche la ligne que si elle a été remplie avec de vrais
        # résultats (le fichier livré par défaut contient des valeurs null
        # tant que le notebook n'a pas été exécuté).
        if transformer_ref.get("accuracy") is not None:
            results.append(transformer_ref)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return results