import pandas as pd
import streamlit as st


def detect_suspicious_patterns(subject: str, body: str) -> dict:
    """
    Detect suspicious or adversarial patterns in a support ticket.

    This is a rule-based detector designed for explainability.
    """
    text = f"{subject} {body}".lower()

    suspicious_keywords = {
        "credential_request": [
            "password",
            "login",
            "credentials",
            "verify your account",
            "confirm your account",
            "security code",
            "one-time code",
            "otp",
            "reset your password",
        ],
        "urgency_pressure": [
            "urgent",
            "immediately",
            "as soon as possible",
            "within 24 hours",
            "final notice",
            "act now",
            "deadline today",
            "account will be locked",
        ],
        "payment_or_invoice": [
            "invoice",
            "payment",
            "bank account",
            "wire transfer",
            "billing",
            "refund",
            "pay now",
            "overdue",
        ],
        "impersonation": [
            "ceo",
            "manager",
            "director",
            "executive",
            "admin team",
            "it department",
            "support team",
        ],
        "suspicious_links": [
            "http://",
            "https://",
            "bit.ly",
            "tinyurl",
            "click here",
            "open this link",
            "download attachment",
        ],
        "threat_language": [
            "account suspended",
            "account locked",
            "legal action",
            "unauthorized access",
            "security breach",
            "compromised account",
        ],
    }

    detected_categories = []
    detected_terms = []

    for category, keywords in suspicious_keywords.items():
        matches = [keyword for keyword in keywords if keyword in text]

        if matches:
            detected_categories.append(category)
            detected_terms.extend(matches)

    risk_score = min(len(detected_categories) * 20, 100)

    if risk_score >= 70:
        risk_level = "High"
    elif risk_score >= 40:
        risk_level = "Medium"
    elif risk_score >= 20:
        risk_level = "Low"
    else:
        risk_level = "Minimal"

    recommended_action = {
        "High": "Do not auto-route only. Escalate to security review before responding.",
        "Medium": "Route normally but flag for manual review.",
        "Low": "Allow normal triage but show warning indicators.",
        "Minimal": "No suspicious pattern detected by the rule-based scanner.",
    }

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "detected_categories": detected_categories,
        "detected_terms": sorted(set(detected_terms)),
        "recommended_action": recommended_action[risk_level],
    }


def display_suspicious_detection_result(result: dict):
    """
    Display suspicious ticket detection results in Streamlit.
    """
    risk_level = result["risk_level"]
    risk_score = result["risk_score"]

    if risk_level == "High":
        st.error(f"Suspicious Risk Level: {risk_level} ({risk_score}/100)")
    elif risk_level == "Medium":
        st.warning(f"Suspicious Risk Level: {risk_level} ({risk_score}/100)")
    elif risk_level == "Low":
        st.info(f"Suspicious Risk Level: {risk_level} ({risk_score}/100)")
    else:
        st.success(f"Suspicious Risk Level: {risk_level} ({risk_score}/100)")

    st.write("**Recommended action:**")
    st.write(result["recommended_action"])

    if result["detected_categories"]:
        st.write("**Detected suspicious categories:**")
        st.write(", ".join(result["detected_categories"]))

    if result["detected_terms"]:
        st.write("**Detected terms:**")
        st.write(", ".join(result["detected_terms"]))


@st.cache_data
def create_suspicious_ticket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the suspicious pattern detector to a sample of tickets.
    """
    results = []

    for _, row in df.iterrows():
        detection = detect_suspicious_patterns(
            row["subject"],
            row["body"],
        )

        results.append(
            {
                "subject": row["subject"],
                "type": row["type"],
                "queue": row["queue"],
                "priority": row["priority"],
                "risk_score": detection["risk_score"],
                "risk_level": detection["risk_level"],
                "detected_categories": ", ".join(detection["detected_categories"]),
                "detected_terms": ", ".join(detection["detected_terms"]),
            }
        )

    return pd.DataFrame(results)