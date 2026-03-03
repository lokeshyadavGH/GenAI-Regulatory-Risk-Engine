import os
import pandas as pd
from openai import OpenAI

# Automatically read API key from environment
client = OpenAI(api_key="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")

# Load analyzed data
df = pd.read_csv("analyzed_data.csv")

# Count high-risk records
high_risk_count = df[df["risk_flag"] == "High Risk"].shape[0]

prompt = f"""
We analyzed a regulatory loan dataset.
There are {high_risk_count} high-risk anomalies detected.

Explain what this means for compliance teams.
What risks should they investigate?
Provide a professional executive summary.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- AI GENERATED SUMMARY ---\n")
print(response.choices[0].message.content)
