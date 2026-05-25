import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
def train_priority_model(df: pd.DataFrame):
    """
    Train a machine learning model to predict ticket priority from ticket text.
    """
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

    return model, accuracy, report, cm, labels


@st.cache_resource
def train_queue_model(df: pd.DataFrame):
    """
    Train a machine learning model to predict the correct support queue.
    """
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