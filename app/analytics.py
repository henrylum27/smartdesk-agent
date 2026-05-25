import pandas as pd
import streamlit as st


def estimate_time_savings(
    ticket_count: int,
    minutes_saved_per_ticket: int = 5,
) -> float:
    """
    Estimate employee hours saved if repetitive tickets are partially automated.
    """
    total_minutes_saved = ticket_count * minutes_saved_per_ticket
    return round(total_minutes_saved / 60, 2)


@st.cache_data
def create_productivity_summary(
    df: pd.DataFrame,
    minutes_saved_per_ticket: int = 5,
) -> pd.DataFrame:
    """
    Create a productivity summary by ticket type.
    """
    summary = (
        df.groupby("type")
        .agg(
            ticket_count=("type", "count"),
            unique_queues=("queue", "nunique"),
            priority_levels=("priority", "nunique"),
        )
        .reset_index()
        .sort_values("ticket_count", ascending=False)
    )

    summary["estimated_hours_saved"] = summary["ticket_count"].apply(
        lambda x: estimate_time_savings(x, minutes_saved_per_ticket)
    )

    summary["automation_potential"] = pd.cut(
        summary["ticket_count"],
        bins=[0, 100, 500, 1000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
    )

    return summary


@st.cache_data
def create_queue_workload_summary(
    df: pd.DataFrame,
    minutes_saved_per_ticket: int = 5,
) -> pd.DataFrame:
    """
    Create workload summary by support queue.
    """
    summary = (
        df.groupby("queue")
        .agg(
            ticket_count=("queue", "count"),
            ticket_types=("type", "nunique"),
            priority_levels=("priority", "nunique"),
        )
        .reset_index()
        .sort_values("ticket_count", ascending=False)
    )

    summary["estimated_hours_saved"] = summary["ticket_count"].apply(
        lambda x: estimate_time_savings(x, minutes_saved_per_ticket)
    )

    return summary