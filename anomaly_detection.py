import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the dataset
df = pd.read_csv("regulatory_data.csv")

# Select numeric columns for analysis
features = df[["loan_amount", "applicant_income"]]

# Create anomaly detection model
model = IsolationForest(contamination=0.05, random_state=42)

df["anomaly"] = model.fit_predict(features)

# Mark anomalies clearly
df["risk_flag"] = df["anomaly"].apply(lambda x: "High Risk" if x == -1 else "Normal")

# Save new file
df.to_csv("analyzed_data.csv", index=False)

print("Anomaly detection completed successfully!")