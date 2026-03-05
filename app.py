import streamlit as st
import pandas as pd
from openai import OpenAI

import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("GenAI Regulatory Risk Analyzer")

# Load data
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Generate synthetic data
np.random.seed(42)

data = {
    "loan_id": range(1, 501),
    "loan_amount": np.random.randint(50000, 500000, 500),
    "applicant_income": np.random.randint(20000, 200000, 500),
    "approval_status": np.random.choice(["Approved", "Rejected"], 500),
    "region": np.random.choice(["North", "South", "East", "West"], 500)
}

df = pd.DataFrame(data)

# Run anomaly detection
features = df[["loan_amount", "applicant_income"]]

model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(features)

df["risk_flag"] = df["anomaly"].apply(lambda x: "High Risk" if x == -1 else "Normal")
st.subheader("Dataset Preview")
st.write(df.head())

# Show risk counts
st.subheader("Risk Distribution")
st.bar_chart(df["risk_flag"].value_counts())

# Generate AI Summary
if st.button("Generate AI Compliance Summary"):

    high_risk_count = df[df["risk_flag"] == "High Risk"].shape[0]

    prompt = f"""
    We analyzed a regulatory loan dataset.
    There are {high_risk_count} high-risk anomalies detected.
    Provide a professional executive compliance summary.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("AI Compliance Summary")
    st.write(response.choices[0].message.content)
