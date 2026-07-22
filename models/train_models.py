"""
models/train_models.py — Entraînement et comparaison de modèles
==================================================================
Entraîne 4 classifieurs classiques sur le dataset IMDB (Logistic
Regression, Naive Bayes, Linear SVM, Random Forest), les compare sur
Accuracy / Precision / Recall / F1 / temps d'entraînement, sélectionne
automatiquement le meilleur et le sérialise avec Joblib pour un
rechargement instantané par l'API (plus de ré-entraînement à chaud).

Usage :
    python -m models.train_models
"""

import time
import json
from collections import Counter
from datetime import datetime, timezone

import joblib
import pandas as pd
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import config
from utils.text_preprocessing import clean_series


# ── 1. Données ────────────────────────────────────────────────────────────
def load_and_prepare_data():
    """
    Télécharge le dataset IMDB (Hugging Face), échantillonne, nettoie le
    texte et effectue le split train/test. Reprend la méthodologie du
    rapport (80/20, random_state=42).
    """
    print("Téléchargement du dataset IMDB...")
    dataset = load_dataset(config.DATASET_NAME)
    # Le split "train" et le split "test" HuggingFace sont tous les deux
    # labellisés (25 000 avis chacun) ; on les concatène pour retrouver les
    # 50 000 avis labellisés du dataset IMDB original, puis on refait notre
    # propre split 80/20 juste en dessous. Le split "unsupervised" (50 000
    # avis, label=-1) n'est pas utilisé ici.
    raw_df = pd.concat(
        [pd.DataFrame(dataset["train"]), pd.DataFrame(dataset["test"])],
        ignore_index=True,
    )
    raw_df["sentiment"] = raw_df["label"].map({0: "negative", 1: "positive"})

    if config.SAMPLE_SIZE:
        raw_df = raw_df.sample(
            config.SAMPLE_SIZE, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

    print("Nettoyage du texte (pipeline NLP)...")
    raw_df["cleaned_text"] = clean_series(raw_df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        raw_df["cleaned_text"],
        raw_df["sentiment"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=raw_df["sentiment"],
    )
    return X_train, X_test, y_train, y_test, raw_df


def compute_and_save_eda(raw_df: pd.DataFrame):
    """
    Calcule les statistiques exploratoires du dataset (distribution des
    classes, mots les plus fréquents globalement et par polarité,
    longueur moyenne des avis) et les sauvegarde pour le dashboard.
    Reprend l'analyse fréquentielle du rapport (section 3.2 / 4.2).
    """
    sentiment_counts = raw_df["sentiment"].value_counts().to_dict()

    all_words = " ".join(raw_df["cleaned_text"]).split()
    top20_words = Counter(all_words).most_common(20)

    positive_words = " ".join(
        raw_df.loc[raw_df["sentiment"] == "positive", "cleaned_text"]
    ).split()
    negative_words = " ".join(
        raw_df.loc[raw_df["sentiment"] == "negative", "cleaned_text"]
    ).split()

    eda = {
        "sentiment_counts": sentiment_counts,
        "top20_words": [{"word": w, "count": c} for w, c in top20_words],
        "top10_positive": [
            {"word": w, "count": c} for w, c in Counter(positive_words).most_common(10)
        ],
        "top10_negative": [
            {"word": w, "count": c} for w, c in Counter(negative_words).most_common(10)
        ],
        "total_samples": len(raw_df),
        "avg_words": round(raw_df["cleaned_text"].apply(lambda x: len(x.split())).mean(), 1),
    }

    with open(config.EDA_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2, ensure_ascii=False)

    print(f"Données EDA sauvegardées -> {config.EDA_DATA_PATH}")
    return eda


# ── 2. Registre des modèles ─────────────────────────────────────────────
def get_model_registry():
    """Dictionnaire {nom affiché: instance de classifieur non entraînée}."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(max_iter=2000, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=50,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }


def build_pipeline(classifier):
    """Encapsule TF-IDF + classifieur dans un seul Pipeline Scikit-Learn
    (évite toute fuite de données, cf. rapport section 5.4)."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.TFIDF_MAX_FEATURES,
                    ngram_range=config.TFIDF_NGRAM_RANGE,
                ),
            ),
            ("clf", classifier),
        ]
    )


# ── 3. Entraînement + évaluation ────────────────────────────────────────
def evaluate_model(name, classifier, X_train, X_test, y_train, y_test):
    pipeline = build_pipeline(classifier)

    start = time.time()
    pipeline.fit(X_train, y_train)
    training_time = round(time.time() - start, 2)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label="positive"
    )
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])

    metrics = {
        "model": name,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "training_time_sec": training_time,
        "precision_positive": round(report["positive"]["precision"] * 100, 1),
        "recall_positive": round(report["positive"]["recall"] * 100, 1),
        "precision_negative": round(report["negative"]["precision"] * 100, 1),
        "recall_negative": round(report["negative"]["recall"] * 100, 1),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }
    return pipeline, metrics


def train_and_compare(X_train, X_test, y_train, y_test):
    """Entraîne les 4 modèles et renvoie (pipelines entraînés, tableau de métriques)."""
    registry = get_model_registry()
    results = []
    pipelines = {}

    for name, clf in registry.items():
        print(f"\nEntraînement : {name}...")
        pipeline, metrics = evaluate_model(
            name, clf, X_train, X_test, y_train, y_test
        )
        pipelines[name] = pipeline
        results.append(metrics)
        print(
            f"  -> Accuracy={metrics['accuracy']}%  F1={metrics['f1_score']}%  "
            f"({metrics['training_time_sec']}s)"
        )

    return pipelines, results


# ── 4. Sélection et sauvegarde ──────────────────────────────────────────
def select_best_model(results, pipelines, metric="f1_score"):
    """Le meilleur modèle est celui avec le F1-score le plus élevé
    (plus robuste que l'accuracy seule sur des classes potentiellement
    déséquilibrées)."""
    best_metrics = max(results, key=lambda r: r[metric])
    best_name = best_metrics["model"]
    return best_name, pipelines[best_name], best_metrics


def save_artifacts(best_name, best_pipeline, results, best_metrics, n_train, n_test):
    joblib.dump(best_pipeline, config.BEST_MODEL_PATH)

    with open(config.COMPARISON_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    metadata = {
        "best_model": best_name,
        "best_metrics": best_metrics,
        "sample_size": config.SAMPLE_SIZE,
        "train_size": n_train,
        "test_size": n_test,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMeilleur modèle : {best_name} (F1={best_metrics['f1_score']}%)")
    print(f"Sauvegardé -> {config.BEST_MODEL_PATH}")
    print(f"Comparatif -> {config.COMPARISON_RESULTS_PATH}")
    print(f"Métadonnées -> {config.MODEL_METADATA_PATH}")


# ── 5. Point d'entrée ────────────────────────────────────────────────────
def main():
    X_train, X_test, y_train, y_test, raw_df = load_and_prepare_data()
    compute_and_save_eda(raw_df)
    pipelines, results = train_and_compare(X_train, X_test, y_train, y_test)
    best_name, best_pipeline, best_metrics = select_best_model(results, pipelines)
    save_artifacts(
        best_name,
        best_pipeline,
        results,
        best_metrics,
        n_train=len(X_train),
        n_test=len(X_test),
    )


if __name__ == "__main__":
    main()