"""
database/history_db.py — Historique des prédictions (SQLite)
================================================================
Persiste chaque prédiction (texte, résultat, confiance, date) et
expose les fonctions nécessaires au dashboard : historique paginé
et statistiques agrégées (total, % positif, % négatif).
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import config


@contextmanager
def get_connection():
    """Ouvre une connexion SQLite avec accès aux colonnes par nom
    (sqlite3.Row) et la ferme proprement même en cas d'erreur."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Crée la table `predictions` si elle n'existe pas encore.
    À appeler une fois au démarrage de l'application."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                cleaned_text TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_prediction(text: str, cleaned_text: str, prediction: str, confidence: float) -> int:
    """Enregistre une prédiction et renvoie l'id de la ligne créée."""
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (text, cleaned_text, prediction, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (text, cleaned_text, prediction, confidence, created_at),
        )
        return cursor.lastrowid


def get_history(limit: int = 50) -> list[dict]:
    """Renvoie les `limit` dernières prédictions, les plus récentes en premier."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, text, prediction, confidence, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_stats() -> dict:
    """Statistiques agrégées pour le dashboard : nombre total de
    prédictions et répartition positive/négative en pourcentage."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
        if total == 0:
            return {"total_predictions": 0, "positive_pct": 0.0, "negative_pct": 0.0}

        positive = conn.execute(
            "SELECT COUNT(*) AS n FROM predictions WHERE prediction = 'positive'"
        ).fetchone()["n"]
        negative = total - positive

        return {
            "total_predictions": total,
            "positive_pct": round(positive / total * 100, 1),
            "negative_pct": round(negative / total * 100, 1),
        }


def clear_history():
    """Vide l'historique (utile pour les tests / réinitialisation manuelle)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions")
