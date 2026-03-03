import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "loan_id": range(1, 501),
    "loan_amount": np.random.randint(50000, 500000, 500),
    "applicant_income": np.random.randint(20000, 200000, 500),
    "approval_status": np.random.choice(["Approved", "Rejected"], 500),
    "region": np.random.choice(["North", "South", "East", "West"], 500)
}

df = pd.DataFrame(data)

df.to_csv("regulatory_data.csv", index=False)

print("Dataset created successfully!")
