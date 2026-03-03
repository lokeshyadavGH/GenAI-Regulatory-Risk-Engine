import streamlit as st
import pandas as pd
from openai import OpenAI

import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("GenAI Regulatory Risk Analyzer")

# Load data
df = pd.read_csv("analyzed_data.csv")

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