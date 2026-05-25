import pandas as pd
import streamlit as st
from datasets import load_dataset


@st.cache_data
def load_ticket_data(sample_size: int = 5000) -> pd.DataFrame:
    """
    Load a sample of the customer support ticket dataset from Hugging Face.
    """
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    df = dataset.to_pandas()

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)

    df = df.dropna(subset=["subject", "body", "priority", "queue", "type"])
    df["ticket_text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

    return df