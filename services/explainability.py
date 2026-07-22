"""
services/explainability.py — Explicabilité des prédictions (LIME)
=====================================================================
Utilise LIME (Local Interpretable Model-agnostic Explanations) pour
identifier les mots qui ont le plus influencé la décision du modèle
sur un avis donné, avec leur poids signé :
    excellent  +0.42
    boring     -0.61

Fonctionne avec n'importe quel classifieur du pipeline (avec ou sans
predict_proba) grâce à un wrapper de probabilité générique.
"""

import numpy as np
from lime.lime_text import LimeTextExplainer

import config

CLASS_NAMES = ["negative", "positive"]


def get_predict_proba_fn(pipeline):
    """
    Fonction texte -> probabilités par classe, utilisable par LIME
    comme par le service de prédiction (confiance affichée à l'utilisateur).
    Certains modèles (LinearSVC) n'exposent pas predict_proba : on
    reconstruit une pseudo-probabilité via une sigmoïde appliquée au
    score de décision. Les autres modèles (Logistic Regression, Naive
    Bayes, Random Forest) utilisent directement predict_proba.
    """
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        return pipeline.predict_proba

    def proba_fn(texts):
        scores = pipeline.decision_function(texts)
        proba_positive = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - proba_positive, proba_positive])

    return proba_fn


def explain_prediction(pipeline, cleaned_text: str, num_features: int = None, num_samples: int = None) -> list[dict]:
    """
    Explique une prédiction : renvoie la liste des mots les plus
    influents avec leur poids signé (positif = pousse vers "positive",
    négatif = pousse vers "negative"), triée par importance décroissante.

    Args:
        pipeline: pipeline Scikit-Learn entraîné (TF-IDF + classifieur)
        cleaned_text: texte déjà nettoyé (même prétraitement que l'entraînement)
        num_features: nombre de mots à retourner (défaut: config.LIME_NUM_FEATURES)
        num_samples: nombre de perturbations LIME (défaut: config.LIME_NUM_SAMPLES)

    Returns:
        [{"word": "excellent", "weight": 0.42}, {"word": "boring", "weight": -0.61}, ...]
    """
    if not cleaned_text.strip():
        return []

    num_features = num_features or config.LIME_NUM_FEATURES
    num_samples = num_samples or config.LIME_NUM_SAMPLES

    explainer = LimeTextExplainer(class_names=CLASS_NAMES)
    proba_fn = get_predict_proba_fn(pipeline)

    explanation = explainer.explain_instance(
        cleaned_text,
        proba_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(1,),  # on explique la classe "positive" (index 1) ; poids négatif = pousse vers "negative"
    )

    word_weights = explanation.as_list(label=1)
    word_weights.sort(key=lambda pair: abs(pair[1]), reverse=True)

    return [{"word": str(word), "weight": round(float(weight), 2)} for word, weight in word_weights]
