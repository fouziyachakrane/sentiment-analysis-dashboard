"""
utils/text_preprocessing.py — Pipeline de nettoyage NLP
==========================================================
Reprend exactement la logique de nettoyage du notebook original
(suppression HTML, minuscules, ponctuation, stop words), mais isolée
dans un module réutilisable par l'entraînement, l'API et les tests.

Correction : la liste de stop words NLTK inclut des mots de négation
("not", "no", "don", "isn"...) qui inversent le sens d'une phrase
("not good" != "good"). Ces mots sont désormais exclus du filtrage.
Note : la regex de nettoyage supprime les apostrophes, donc "don't"
devient "dont" avant le filtrage — la liste ci-dessous couvre les
formes sans apostrophe.
"""

import re
import nltk
from nltk.corpus import stopwords
from functools import lru_cache


NEGATION_WORDS = {
    "no", "not", "nor", "never",
    "don", "dont", "doesn", "doesnt", "didn", "didnt",
    "isn", "isnt", "aren", "arent", "wasn", "wasnt", "werent",
    "hasn", "hasnt", "haven", "havent", "hadn", "hadnt",
    "won", "wont", "wouldn", "wouldnt",
    "can", "cant", "couldn", "couldnt",
    "shouldn", "shouldnt", "ain",
}


@lru_cache(maxsize=1)
def get_stopwords() -> set:
    """
    Télécharge (si nécessaire) et met en cache le corpus de stop words
    anglais de NLTK, en excluant les mots de négation (ex: "not", "no",
    "dont", "isnt"...). Ces mots sont indispensables au sens de la
    phrase et ne doivent jamais être supprimés. lru_cache évite de
    recharger le corpus à chaque appel.
    """
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english")) - NEGATION_WORDS


def clean_text(text: str) -> str:
    """
    Nettoie un texte brut selon le pipeline NLP défini dans le rapport :
      1. Suppression des balises HTML (ex: <br />)
      2. Normalisation en minuscules
      3. Suppression de la ponctuation / caractères non alphabétiques
      4. Suppression des espaces multiples
      5. Suppression des stop words anglais (hors mots de négation)

    Args:
        text: texte brut (avis client)

    Returns:
        Texte nettoyé, mots séparés par des espaces. Chaîne vide si
        l'entrée n'est pas exploitable.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = re.sub(r"<.*?>", " ", text)                 # balises HTML
    text = text.lower()                                  # minuscules
    text = re.sub(r"[^a-z\s]", "", text)                 # ponctuation / chiffres
    text = re.sub(r"\s+", " ", text).strip()             # espaces multiples

    stop_words = get_stopwords()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


def clean_series(series):
    """
    Applique clean_text() à une colonne pandas (Series) de textes bruts.
    Utile pour le prétraitement en masse pendant l'entraînement.
    """
    return series.apply(clean_text)