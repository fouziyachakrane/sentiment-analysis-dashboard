"""
app.py — Application Flask : API + service du frontend
==========================================================
Point d'entrée unique de l'application. Au démarrage, initialise la
base SQLite et s'assure qu'un modèle entraîné existe (sinon indique
comment le générer). Le modèle n'est plus ré-entraîné à chaque
démarrage : il est chargé une fois depuis saved_models/best_model.joblib.

Lancement :
    python app.py
Accès :
    http://localhost:5000
"""

import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
from database import history_db
from services import prediction_service, eda_service

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


# ── Démarrage ────────────────────────────────────────────────────────────
def init_app():
    history_db.init_db()

    if not os.path.exists(config.BEST_MODEL_PATH):
        print(
            "\n[ATTENTION] Aucun modèle entraîné trouvé.\n"
            "Lancez d'abord : python -m models.train_models\n"
        )
    else:
        prediction_service.load_model()  # pré-charge le modèle en cache
        print(f"Modèle chargé depuis {config.BEST_MODEL_PATH}")


# ── Frontend ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── API : Prédiction ────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Entrée : {"text": "Amazing movie"}
    Sortie : {"prediction": "Positive", "confidence": 98.2, ...}
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    try:
        result = prediction_service.predict_sentiment(text, with_explanation=False)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Modèle non entraîné. Lancez models/train_models.py."}), 503

    return jsonify(
        {
            "prediction": result["prediction"].capitalize(),
            "confidence": result["confidence"],
            "original_text": result["original_text"],
            "cleaned_text": result["cleaned_text"],
        }
    )


@app.route("/api/explain", methods=["POST"])
def explain():
    """
    Entrée : {"text": "Amazing movie"}
    Sortie : prédiction + confiance + mots les plus influents (LIME)
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    try:
        result = prediction_service.predict_sentiment(
            text, with_explanation=True, save_to_history=False
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Modèle non entraîné. Lancez models/train_models.py."}), 503

    return jsonify(
        {
            "prediction": result["prediction"].capitalize(),
            "confidence": result["confidence"],
            "cleaned_text": result["cleaned_text"],
            "explanation": result["explanation"],
        }
    )


# ── API : Dashboard ──────────────────────────────────────────────────────
@app.route("/api/eda")
def get_eda():
    try:
        return jsonify(eda_service.get_dashboard_eda())
    except FileNotFoundError:
        return jsonify({"error": "Modèle non entraîné. Lancez models/train_models.py."}), 503


@app.route("/api/stats")
def get_stats():
    """Statistiques des prédictions utilisateur (issues de l'historique SQLite)."""
    return jsonify(history_db.get_stats())


@app.route("/api/history")
def get_history():
    limit = request.args.get("limit", default=50, type=int)
    return jsonify(history_db.get_history(limit=limit))


@app.route("/api/compare-models")
def compare_models():
    """Tableau comparatif des 4 modèles entraînés + meilleur modèle retenu."""
    try:
        results = prediction_service.get_comparison_results()
        metadata = prediction_service.get_model_metadata()
    except FileNotFoundError:
        return jsonify({"error": "Modèle non entraîné. Lancez models/train_models.py."}), 503

    return jsonify({"models": results, "best_model": metadata["best_model"]})


# ── Erreurs génériques ───────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route non trouvée"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Erreur interne du serveur"}), 500


if __name__ == "__main__":
    init_app()
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG_MODE)
