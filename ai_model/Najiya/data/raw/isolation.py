import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# 1️⃣ Load the combined dataset
input_file = "/home/najiya/fraud shipment identification/data/processed/combined_shipments.parquet"
df = pd.read_parquet(input_file)
print(f"Loaded dataset with shape: {df.shape}")

# 2️⃣ Select numeric features for anomaly detection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Using numeric columns: {numeric_cols}")

X = df[numeric_cols]

# 3️⃣ Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Train Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # assumes ~5% anomalies, tweak if needed
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled)

# 5️⃣ Predict anomaly scores
df["anomaly_score"] = model.decision_function(X_scaled)
df["is_anomaly"] = model.predict(X_scaled)  # -1 = anomaly, 1 = normal

# 6️⃣ Save results
output_file = "/home/najiya/fraud shipment identification/data/processed/isolation_forest_results.parquet"
df.to_parquet(output_file, index=False)
print(f"✅ Anomaly detection completed. Results saved to {output_file}")

# 7️⃣ Show quick summary
print(df["is_anomaly"].value_counts())
print(df[["anomaly_score", "is_anomaly"]].head())
