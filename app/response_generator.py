def summarize_ticket(subject: str, body: str) -> dict:
    """
    Create an explainable ticket summary without using a paid LLM API.
    """
    text = f"{subject} {body}".lower()

    issue_keywords = {
        "Login / Access Issue": [
            "login",
            "log in",
            "access",
            "password",
            "account",
            "locked",
            "sign in",
        ],
        "Payment / Billing Issue": [
            "payment",
            "invoice",
            "billing",
            "refund",
            "charge",
            "bank",
            "subscription",
        ],
        "Technical Bug": [
            "error",
            "bug",
            "crash",
            "not working",
            "failed",
            "broken",
            "issue",
        ],
        "System Outage": [
            "down",
            "outage",
            "unavailable",
            "all users",
            "system is down",
        ],
        "Request / Account Change": [
            "change",
            "update",
            "request",
            "modify",
            "remove",
            "add",
        ],
        "Security Concern": [
            "security",
            "phishing",
            "unauthorized",
            "breach",
            "compromised",
            "suspicious",
        ],
    }

    detected_issue = "General Support Request"

    for issue_type, keywords in issue_keywords.items():
        if any(keyword in text for keyword in keywords):
            detected_issue = issue_type
            break

    impact = "Normal business impact"

    if any(
        word in text
        for word in ["urgent", "immediately", "deadline", "all users", "unable to work"]
    ):
        impact = "High business impact"

    if any(
        word in text
        for word in [
            "cannot access",
            "unable to access",
            "system is down",
            "payment system",
        ]
    ):
        impact = "Operational disruption"

    summary = (
        f"The ticket appears to be a {detected_issue.lower()} involving: "
        f"{subject.strip()}."
    )

    recommended_next_step = get_recommended_next_step(detected_issue)

    return {
        "summary": summary,
        "issue_type": detected_issue,
        "impact": impact,
        "recommended_next_step": recommended_next_step,
    }


def get_recommended_next_step(issue_type: str) -> str:
    """
    Recommend a support next step based on detected issue type.
    """
    recommendations = {
        "Login / Access Issue": "Verify the user's identity, check account status, and guide them through access recovery.",
        "Payment / Billing Issue": "Review billing records, confirm account details, and route to the billing team if needed.",
        "Technical Bug": "Collect reproduction steps, check logs, and escalate to the technical support team.",
        "System Outage": "Escalate immediately, check service status, and communicate expected resolution updates.",
        "Request / Account Change": "Confirm the requested change, verify permissions, and update the account record.",
        "Security Concern": "Do not process automatically. Escalate to security review before taking action.",
        "General Support Request": "Review the ticket details and route it to the appropriate support team.",
    }

    return recommendations.get(issue_type, recommendations["General Support Request"])


def draft_support_response(
    subject: str,
    body: str,
    priority_prediction: str,
    queue_prediction: str,
    suspicious_result: dict,
    ticket_summary: dict,
) -> str:
    """
    Draft a professional support response using ticket analysis outputs.
    """
    issue_type = ticket_summary["issue_type"]
    next_step = ticket_summary["recommended_next_step"]
    risk_level = suspicious_result["risk_level"]

    if risk_level in ["High", "Medium"]:
        return f"""
Hello,

Thank you for contacting support. We have received your request regarding "{subject}".

For your safety, this ticket has been flagged for additional review because it contains patterns that may require verification before we proceed.

Our team will review the request carefully before taking action.

Ticket assessment:
- Detected issue type: {issue_type}
- Suggested priority: {priority_prediction}
- Suggested routing queue: {queue_prediction}
- Security review level: {risk_level}

Recommended internal next step:
{next_step}

Kind regards,  
SmartDesk Support Team
"""

    return f"""
Hello,

Thank you for contacting support. We have received your request regarding "{subject}".

Based on the ticket details, this appears to be a {issue_type.lower()}. Your request has been routed to the recommended support queue for review.

Ticket assessment:
- Suggested priority: {priority_prediction}
- Suggested routing queue: {queue_prediction}
- Expected next step: {next_step}

A support team member will review the ticket and follow up with the appropriate action.

Kind regards,  
SmartDesk Support Team
"""