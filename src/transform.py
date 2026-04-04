import pandas as pd
import spacy
import json
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.db import get_engine

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    text = ' '.join(text.split())
    return text

def extract_entities(text):
    doc = nlp(text[:1000])
    entities = [ent.text for ent in doc.ents]
    return json.dumps(entities)

def transform_and_train():
    print("Reading raw notes from MySQL...")
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM raw_notes", con=engine)
    print(f"Loaded {len(df)} rows")

    print("Cleaning text...")
    df["cleaned_text"] = df["transcription"].apply(clean_text)

    print("Extracting entities with spaCy (this may take a few minutes)...")
    df["entities"] = df["cleaned_text"].apply(extract_entities)

    print("Training classifier...")
    X = df["cleaned_text"]
    y = df["medical_specialty"].str.strip()

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("\nModel Performance:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Saving model and vectorizer...")
    joblib.dump(model, "models/classifier.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Writing processed notes to MySQL...")
    df["predicted_specialty"] = model.predict(X_vectorized)
    proba = model.predict_proba(X_vectorized)
    df["confidence_score"] = proba.max(axis=1).round(4)

    processed_df = df[[
        "id", "cleaned_text", "entities",
        "predicted_specialty", "confidence_score"
    ]].copy()

    processed_df.to_sql("processed_notes", con=engine,
                        if_exists="replace", index=False)
    print(f"Done. {len(processed_df)} rows written to processed_notes table")

if __name__ == "__main__":
    transform_and_train()