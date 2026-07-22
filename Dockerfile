FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Corpus NLTK téléchargé au build (évite le téléchargement à chaque démarrage)
RUN python -c "import nltk; nltk.download('stopwords')"

COPY . .

# Entraîne le modèle au build si aucun modèle n'est déjà présent dans saved_models/
RUN python -m models.train_models || echo "Entraînement ignoré (à faire manuellement si besoin)"

EXPOSE 7860
ENV PORT=7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120"]
