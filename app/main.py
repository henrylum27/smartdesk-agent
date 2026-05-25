import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_ticket_data
from analytics import create_productivity_summary, create_queue_workload_summary
from ml_models import train_priority_model, train_queue_model, predict_with_confidence
from risk_detector import (
    create_suspicious_ticket_summary,
    detect_suspicious_patterns,
    display_suspicious_detection_result,
)
from response_generator import summarize_ticket, draft_support_response


st.set_page_config(
    page_title="SmartDesk Agent",
    page_icon="🎧",
    layout="wide",
)


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
            title="Tickets by Type",
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with right_col:
        priority_counts = df["priority"].value_counts().reset_index()
        priority_counts.columns = ["priority", "count"]

        fig_priority = px.bar(
            priority_counts,
            x="priority",
            y="count",
            title="Tickets by Priority",
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
        title="Top 15 Support Queues",
    )
    st.plotly_chart(fig_queue, use_container_width=True)

    # -----------------------------
    # Productivity Insights
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
            f"{top_repetitive_type['ticket_count']:,} tickets",
        )

    with col2:
        st.metric(
            "Highest Workload Queue",
            top_queue["queue"],
            f"{top_queue['ticket_count']:,} tickets",
        )

    with col3:
        st.metric(
            "Estimated Automation Savings",
            f"{total_estimated_hours_saved:,.1f} hours",
            "Assuming 5 min saved per ticket",
        )

    st.subheader("Repetitive Ticket Types")

    fig_repetitive = px.bar(
        productivity_df.head(10),
        x="ticket_count",
        y="type",
        orientation="h",
        title="Top 10 Most Repetitive Ticket Types",
        hover_data=["estimated_hours_saved", "automation_potential"],
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
                "automation_potential",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Queue Workload Hotspots")

    fig_queue_workload = px.bar(
        queue_workload_df.head(15),
        x="ticket_count",
        y="queue",
        orientation="h",
        title="Queues With Highest Ticket Workload",
        hover_data=["estimated_hours_saved", "ticket_types"],
    )
    st.plotly_chart(fig_queue_workload, use_container_width=True)

    st.dataframe(
        queue_workload_df[
            [
                "queue",
                "ticket_count",
                "ticket_types",
                "priority_levels",
                "estimated_hours_saved",
            ]
        ].head(20),
        use_container_width=True,
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
    # ML Priority Predictor
    # -----------------------------
    st.header("Machine Learning: Ticket Priority Predictor")

    st.write(
        """
        This model predicts ticket priority from the subject and body text.
        It uses TF-IDF text features and logistic regression.
        """
    )

    with st.spinner("Training priority prediction model..."):
        priority_model, priority_accuracy, priority_report, priority_cm, priority_labels = (
            train_priority_model(df)
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Priority Model Accuracy", f"{priority_accuracy:.2%}")

    with col2:
        st.metric("Training Rows Used", f"{len(df):,}")

    st.subheader("Priority Model Performance")

    priority_report_df = (
        pd.DataFrame(priority_report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )

    priority_report_df = priority_report_df[
        priority_report_df["class"].isin(priority_labels)
    ]

    st.dataframe(
        priority_report_df[["class", "precision", "recall", "f1-score", "support"]],
        use_container_width=True,
    )

    st.subheader("Priority Confusion Matrix")

    priority_cm_df = pd.DataFrame(
        priority_cm,
        index=priority_labels,
        columns=priority_labels,
    )

    fig_priority_cm = px.imshow(
        priority_cm_df,
        text_auto=True,
        title="Priority Prediction Confusion Matrix",
        labels=dict(x="Predicted Priority", y="Actual Priority"),
    )
    st.plotly_chart(fig_priority_cm, use_container_width=True)

    # -----------------------------
    # Suspicious / Adversarial Detection
    # -----------------------------
    st.header("Suspicious / Adversarial Ticket Detection")

    st.write(
        """
        This section scans support tickets for suspicious patterns such as phishing,
        social engineering, credential requests, suspicious links, payment scams,
        and urgency pressure.
        """
    )

    with st.spinner("Scanning tickets for suspicious patterns..."):
        suspicious_df = create_suspicious_ticket_summary(df.head(1000))

    risk_counts = suspicious_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tickets Scanned", f"{len(suspicious_df):,}")

    with col2:
        high_risk_count = len(suspicious_df[suspicious_df["risk_level"] == "High"])
        st.metric("High Risk Tickets", f"{high_risk_count:,}")

    with col3:
        review_count = len(
            suspicious_df[suspicious_df["risk_level"].isin(["High", "Medium"])]
        )
        st.metric("Needs Review", f"{review_count:,}")

    fig_risk = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        title="Suspicious Risk Levels in Sample Tickets",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("Highest Risk Tickets")

    high_risk_tickets = suspicious_df.sort_values(
        "risk_score",
        ascending=False,
    ).head(20)

    st.dataframe(
        high_risk_tickets[
            [
                "subject",
                "type",
                "queue",
                "priority",
                "risk_score",
                "risk_level",
                "detected_categories",
                "detected_terms",
            ]
        ],
        use_container_width=True,
    )

    st.info(
        """
        This detector is intentionally explainable and rule-based. It is designed
        as a first safety layer, not a final security decision system.
        Later versions can add a trained phishing classifier.
        """
    )

    # -----------------------------
    # ML Queue Routing
    # -----------------------------
    st.header("Machine Learning: Queue Routing Predictor")

    st.write(
        """
        This model predicts which support queue should handle a ticket.
        This reduces manual triage work by automatically suggesting the right team.
        """
    )

    with st.spinner("Training queue routing model..."):
        queue_model, queue_accuracy, queue_report, queue_cm, queue_labels, queue_training_rows = (
            train_queue_model(df)
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Queue Model Accuracy", f"{queue_accuracy:.2%}")

    with col2:
        st.metric("Queues Learned", f"{len(queue_labels):,}")

    with col3:
        st.metric("Training Rows Used", f"{queue_training_rows:,}")

    st.subheader("Queue Model Performance")

    queue_report_df = (
        pd.DataFrame(queue_report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )

    queue_report_df = queue_report_df[queue_report_df["class"].isin(queue_labels)]

    st.dataframe(
        queue_report_df[["class", "precision", "recall", "f1-score", "support"]],
        use_container_width=True,
    )

    st.subheader("Queue Confusion Matrix")

    st.write(
        """
        The queue model may have many labels, so this matrix can be large.
        Focus on the strongest diagonal values where actual and predicted queues match.
        """
    )

    queue_cm_df = pd.DataFrame(
        queue_cm,
        index=queue_labels,
        columns=queue_labels,
    )

    fig_queue_cm = px.imshow(
        queue_cm_df,
        text_auto=False,
        title="Queue Routing Confusion Matrix",
        labels=dict(x="Predicted Queue", y="Actual Queue"),
    )
    st.plotly_chart(fig_queue_cm, use_container_width=True)

    # -----------------------------
    # Combined Triage Demo
    # -----------------------------
    st.header("Smart Ticket Triage Demo")

    st.write(
        """
        Enter a new support ticket. The app will predict priority, recommend
        a support queue, detect suspicious patterns, summarize the issue,
        and draft a response.
        """
    )

    custom_subject = st.text_input(
        "Ticket subject",
        value="Payment system is down for all users",
    )

    custom_body = st.text_area(
        "Ticket body",
        value=(
            "Customers are unable to complete payments. This is affecting all checkout "
            "transactions and needs immediate attention."
        ),
        height=150,
    )

    if st.button("Analyze Ticket"):
        priority_prediction, priority_confidence_df = predict_with_confidence(
            priority_model,
            custom_subject,
            custom_body,
            "priority",
        )

        queue_prediction, queue_confidence_df = predict_with_confidence(
            queue_model,
            custom_subject,
            custom_body,
            "queue",
        )

        suspicious_result = detect_suspicious_patterns(
            custom_subject,
            custom_body,
        )

        ticket_summary = summarize_ticket(
            custom_subject,
            custom_body,
        )

        response_draft = draft_support_response(
            custom_subject,
            custom_body,
            priority_prediction,
            queue_prediction,
            suspicious_result,
            ticket_summary,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"Predicted Priority: {priority_prediction}")

            fig_priority_confidence = px.bar(
                priority_confidence_df,
                x="priority",
                y="confidence",
                title="Priority Prediction Confidence",
            )
            st.plotly_chart(fig_priority_confidence, use_container_width=True)

        with col2:
            st.success(f"Recommended Queue: {queue_prediction}")

            fig_queue_confidence = px.bar(
                queue_confidence_df.head(10),
                x="queue",
                y="confidence",
                title="Top Queue Routing Confidence Scores",
            )
            st.plotly_chart(fig_queue_confidence, use_container_width=True)

        st.subheader("AI Ticket Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Summary:**")
            st.write(ticket_summary["summary"])

            st.write("**Detected issue type:**")
            st.write(ticket_summary["issue_type"])

        with col2:
            st.write("**Customer / business impact:**")
            st.write(ticket_summary["impact"])

            st.write("**Recommended next step:**")
            st.write(ticket_summary["recommended_next_step"])

        st.subheader("Suspicious / Adversarial Pattern Detection")

        display_suspicious_detection_result(suspicious_result)

        st.subheader("Agent Recommendation")

        if suspicious_result["risk_level"] in ["High", "Medium"]:
            st.write(
                f"""
                **Recommended action:** Route this ticket to **{queue_prediction}** with **{priority_prediction}** priority,
                but flag it for additional review because suspicious patterns were detected.

                **Why this matters:**
                - Reduces manual ticket sorting
                - Helps employees avoid phishing and social engineering attempts
                - Prevents suspicious requests from being handled as normal tickets
                - Supports safer and faster helpdesk operations
                """
            )
        else:
            st.write(
                f"""
                **Recommended action:** Route this ticket to **{queue_prediction}** with **{priority_prediction}** priority.

                **Why this helps productivity:**
                - Reduces manual ticket sorting
                - Speeds up first response time
                - Helps employees focus on resolution instead of triage
                - Creates a repeatable process for high-volume support teams
                """
            )

        st.subheader("Draft Support Response")

        st.text_area(
            "Generated response draft",
            value=response_draft.strip(),
            height=300,
        )

    # -----------------------------
    # Ticket Explorer
    # -----------------------------
    st.header("Sample Ticket Explorer")

    selected_priority = st.selectbox(
        "Filter by priority",
        options=["All"] + sorted(df["priority"].dropna().unique().tolist()),
    )

    filtered_df = df.copy()

    if selected_priority != "All":
        filtered_df = filtered_df[filtered_df["priority"] == selected_priority]

    st.dataframe(
        filtered_df[["subject", "body", "type", "queue", "priority"]].head(50),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()