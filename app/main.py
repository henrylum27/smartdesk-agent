import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import load_dataset


st.set_page_config(
    page_title="SmartDesk Agent",
    page_icon="🎧",
    layout="wide"
)


@st.cache_data
def load_ticket_data(sample_size: int = 5000) -> pd.DataFrame:
    """
    Load a sample of the customer support ticket dataset from Hugging Face.
    The full dataset has about 61.8k rows, but we start with a smaller sample
    to keep the app fast while developing.
    """
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    df = dataset.to_pandas()

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)

    return df


def main():
    st.title("SmartDesk Agent")
    st.subheader("AI Helpdesk Ticket Triage and Threat Detection Assistant")

    st.write(
        """
        This dashboard analyzes support tickets to help teams reduce repetitive work,
        improve productivity, and prepare for automated ticket triage.
        """
    )

    with st.spinner("Loading support ticket dataset..."):
        df = load_ticket_data()

    st.success(f"Loaded {len(df):,} tickets")

    st.header("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tickets", f"{len(df):,}")

    with col2:
        st.metric("Ticket Types", df["type"].nunique())

    with col3:
        st.metric("Support Queues", df["queue"].nunique())

    with col4:
        st.metric("Priority Levels", df["priority"].nunique())

    st.header("Ticket Distribution")

    left_col, right_col = st.columns(2)

    with left_col:
        type_counts = df["type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]

        fig_type = px.bar(
            type_counts,
            x="type",
            y="count",
            title="Tickets by Type"
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with right_col:
        priority_counts = df["priority"].value_counts().reset_index()
        priority_counts.columns = ["priority", "count"]

        fig_priority = px.bar(
            priority_counts,
            x="priority",
            y="count",
            title="Tickets by Priority"
        )
        st.plotly_chart(fig_priority, use_container_width=True)

    st.header("Top Support Queues")

    queue_counts = df["queue"].value_counts().head(15).reset_index()
    queue_counts.columns = ["queue", "count"]

    fig_queue = px.bar(
        queue_counts,
        x="count",
        y="queue",
        orientation="h",
        title="Top 15 Support Queues"
    )
    st.plotly_chart(fig_queue, use_container_width=True)

    st.header("Sample Ticket Explorer")

    selected_priority = st.selectbox(
        "Filter by priority",
        options=["All"] + sorted(df["priority"].dropna().unique().tolist())
    )

    filtered_df = df.copy()

    if selected_priority != "All":
        filtered_df = filtered_df[filtered_df["priority"] == selected_priority]

    st.dataframe(
        filtered_df[["subject", "body", "type", "queue", "priority"]].head(50),
        use_container_width=True
    )


if __name__ == "__main__":
    main()