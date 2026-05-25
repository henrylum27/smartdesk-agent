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
    """
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    df = dataset.to_pandas()

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)

    return df


def estimate_time_savings(ticket_count: int, minutes_saved_per_ticket: int = 5) -> float:
    """
    Estimate employee hours saved if repetitive tickets are partially automated.
    """
    total_minutes_saved = ticket_count * minutes_saved_per_ticket
    return round(total_minutes_saved / 60, 2)


def create_productivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a productivity summary by ticket type.
    """
    summary = (
        df.groupby("type")
        .agg(
            ticket_count=("type", "count"),
            unique_queues=("queue", "nunique"),
            priority_levels=("priority", "nunique")
        )
        .reset_index()
        .sort_values("ticket_count", ascending=False)
    )

    summary["estimated_hours_saved"] = summary["ticket_count"].apply(
        lambda x: estimate_time_savings(x)
    )

    summary["automation_potential"] = pd.cut(
        summary["ticket_count"],
        bins=[0, 100, 500, 1000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"]
    )

    return summary


def create_queue_workload_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create workload summary by support queue.
    """
    summary = (
        df.groupby("queue")
        .agg(
            ticket_count=("queue", "count"),
            ticket_types=("type", "nunique"),
            priority_levels=("priority", "nunique")
        )
        .reset_index()
        .sort_values("ticket_count", ascending=False)
    )

    summary["estimated_hours_saved"] = summary["ticket_count"].apply(
        lambda x: estimate_time_savings(x)
    )

    return summary


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

    # -----------------------------
    # Dataset Preview
    # -----------------------------
    st.header("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # -----------------------------
    # Dataset Overview
    # -----------------------------
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

    # -----------------------------
    # Ticket Distribution
    # -----------------------------
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

    # -----------------------------
    # Version 2: Productivity Insights
    # -----------------------------
    st.header("Productivity Insights")

    productivity_df = create_productivity_summary(df)
    queue_workload_df = create_queue_workload_summary(df)

    top_repetitive_type = productivity_df.iloc[0]
    top_queue = queue_workload_df.iloc[0]
    total_estimated_hours_saved = productivity_df["estimated_hours_saved"].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Most Repetitive Ticket Type",
            top_repetitive_type["type"],
            f"{top_repetitive_type['ticket_count']:,} tickets"
        )

    with col2:
        st.metric(
            "Highest Workload Queue",
            top_queue["queue"],
            f"{top_queue['ticket_count']:,} tickets"
        )

    with col3:
        st.metric(
            "Estimated Automation Savings",
            f"{total_estimated_hours_saved:,.1f} hours",
            "Assuming 5 min saved per ticket"
        )

    st.subheader("Repetitive Ticket Types")

    fig_repetitive = px.bar(
        productivity_df.head(10),
        x="ticket_count",
        y="type",
        orientation="h",
        title="Top 10 Most Repetitive Ticket Types",
        hover_data=["estimated_hours_saved", "automation_potential"]
    )
    st.plotly_chart(fig_repetitive, use_container_width=True)

    st.dataframe(
        productivity_df[
            [
                "type",
                "ticket_count",
                "unique_queues",
                "priority_levels",
                "estimated_hours_saved",
                "automation_potential"
            ]
        ],
        use_container_width=True
    )

    st.subheader("Queue Workload Hotspots")

    fig_queue_workload = px.bar(
        queue_workload_df.head(15),
        x="ticket_count",
        y="queue",
        orientation="h",
        title="Queues With Highest Ticket Workload",
        hover_data=["estimated_hours_saved", "ticket_types"]
    )
    st.plotly_chart(fig_queue_workload, use_container_width=True)

    st.dataframe(
        queue_workload_df[
            [
                "queue",
                "ticket_count",
                "ticket_types",
                "priority_levels",
                "estimated_hours_saved"
            ]
        ].head(20),
        use_container_width=True
    )

    st.subheader("Top Automation Opportunities")

    automation_opportunities = productivity_df[
        productivity_df["automation_potential"].isin(["High", "Very High"])
    ].head(10)

    for _, row in automation_opportunities.iterrows():
        st.write(
            f"""
            **{row['type']}**
            - Ticket volume: {row['ticket_count']:,}
            - Estimated time savings: {row['estimated_hours_saved']:,.1f} hours
            - Automation potential: {row['automation_potential']}
            - Suggested automation: auto-classification, routing, response templates, and FAQ suggestions
            """
        )

    st.info(
        """
        Productivity assumption: this version estimates savings by assuming that automation
        saves 5 minutes per repetitive ticket. Later versions can make this configurable
        and more realistic by using ticket complexity, priority, and queue.
        """
    )

    # -----------------------------
    # Ticket Explorer
    # -----------------------------
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