# SmartDesk Agent

SmartDesk Agent is an AI-powered helpdesk analytics project designed to improve employee productivity, reduce repetitive support tasks, and detect suspicious or adversarial ticket patterns.

## Project Goals

This project focuses on three areas:

1. Productivity gains through automated ticket analysis
2. Reducing repetitive employee tasks through ticket triage and response assistance
3. Using machine learning to detect suspicious or adversarial support messages

## Current Features

- Loads a realistic customer support ticket dataset
- Displays dataset overview
- Shows ticket counts by type, queue, and priority
- Provides a simple ticket explorer
- Prepares the foundation for ML-based ticket classification

## Dataset

This project uses the `Tobi-Bueck/customer-support-tickets` dataset from Hugging Face.

The dataset contains customer support ticket subjects, bodies, answers, ticket types, queues, and priorities.

License: CC BY-NC 4.0

## Tech Stack

- Python
- Streamlit
- pandas
- Plotly
- Hugging Face Datasets
- scikit-learn

## Version 2 Features

- Identifies the most repetitive ticket types
- Calculates queue workload hotspots
- Estimates potential employee time savings from automation
- Ranks ticket types by automation potential
- Suggests automation opportunities such as auto-routing, response templates, and FAQ suggestions

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/main.py