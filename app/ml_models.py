import pandas as pd
import streamlit as st
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
MODEL_DIR = Path("models")
PRIORITY_MODEL_PATH = MODEL_DIR / "priority_model.joblib"
QUEUE_MODEL_PATH = MODEL_DIR / "queue_model.joblib"

def build_text_classifier() -> Pipeline:
    """
    Build a reusable text classification pipeline.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


@st.cache_resource
@st.cache_resource
def train_priority_model(df: pd.DataFrame, force_retrain: bool = False):
    """
    Train or load a machine learning model to predict ticket priority from ticket text.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    if PRIORITY_MODEL_PATH.exists() and not force_retrain:
        saved_data = joblib.load(PRIORITY_MODEL_PATH)
        return (
            saved_data["model"],
            saved_data["accuracy"],
            saved_data["report"],
            saved_data["confusion_matrix"],
            saved_data["labels"],
        )

    X = df["ticket_text"]
    y = df["priority"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_text_classifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    saved_data = {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "labels": labels,
    }

    joblib.dump(saved_data, PRIORITY_MODEL_PATH)

    return model, accuracy, report, cm, labels


@st.cache_resource
def train_queue_model(df: pd.DataFrame, force_retrain: bool = False):
    """
    Train or load a machine learning model to predict the correct support queue.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    if QUEUE_MODEL_PATH.exists() and not force_retrain:
        saved_data = joblib.load(QUEUE_MODEL_PATH)
        return (
            saved_data["model"],
            saved_data["accuracy"],
            saved_data["report"],
            saved_data["confusion_matrix"],
            saved_data["labels"],
            saved_data["training_rows"],
        )

    queue_counts = df["queue"].value_counts()
    valid_queues = queue_counts[queue_counts >= 10].index
    filtered_df = df[df["queue"].isin(valid_queues)].copy()

    X = filtered_df["ticket_text"]
    y = filtered_df["queue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = build_text_classifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    saved_data = {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "training_rows": len(filtered_df),
    }

    joblib.dump(saved_data, QUEUE_MODEL_PATH)

    return model, accuracy, report, cm, labels, len(filtered_df)

def predict_with_confidence(model, subject: str, body: str, label_name: str) -> tuple:
    """
    Predict a label and return class confidence scores.
    """
    ticket_text = subject + " " + body

    prediction = model.predict([ticket_text])[0]
    probabilities = model.predict_proba([ticket_text])[0]
    class_labels = model.classes_

    confidence_df = pd.DataFrame(
        {
            label_name: class_labels,
            "confidence": probabilities,
        }
    ).sort_values("confidence", ascending=False)

    return prediction, confidence_df

def delete_saved_models():
    """
    Delete saved model files so the app can retrain fresh models.
    """
    deleted_files = []

    for model_path in [PRIORITY_MODEL_PATH, QUEUE_MODEL_PATH]:
        if model_path.exists():
            model_path.unlink()
            deleted_files.append(str(model_path))

    return deleted_files