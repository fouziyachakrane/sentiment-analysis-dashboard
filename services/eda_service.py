"""
services/eda_service.py — Données du dashboard
==================================================
Fusionne les statistiques exploratoires du dataset (eda_data.json) et
les métriques du meilleur modèle (metadata.json) en un seul objet
consommé par l'onglet "Dashboard" du frontend.
"""

import json

import config


def get_dashboard_eda() -> dict:
    """
    Renvoie un dictionnaire unique combinant :
      - les statistiques du dataset (distribution, mots fréquents, longueur moyenne)
      - les métriques du meilleur modèle (accuracy, precision/recall par classe)
    """
    with open(config.EDA_DATA_PATH, encoding="utf-8") as f:
        eda = json.load(f)

    with open(config.MODEL_METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    best_metrics = metadata["best_metrics"]

    eda.update(
        {
            "best_model_name": metadata["best_model"],
            "model_accuracy": best_metrics["accuracy"],
            "precision_positive": best_metrics["precision_positive"],
            "recall_positive": best_metrics["recall_positive"],
            "precision_negative": best_metrics["precision_negative"],
            "recall_negative": best_metrics["recall_negative"],
            "confusion_matrix": best_metrics["confusion_matrix"],
        }
    )
    return eda
