import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Config
INPUT_CSV = "test_shipments.csv"     
OUTPUT_CSV = "predicted_fraud_if.csv"
DUMMY_YEAR = "2025"
RANDOM_SEED = 42
PERCENTILE_THRESHOLD = 95  # mark top 5% highest scores as fraud

# Haversine distance
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Parse datetime
for col in ["accept_time", "delivery_time"]:
    if col in df.columns:
        df[col] = pd.to_datetime(
            DUMMY_YEAR + "-" + df[col].astype(str).str.strip(),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )

# Compute features
if set(["accept_time", "delivery_time"]).issubset(df.columns):
    df["delivery_delay"] = (df["delivery_time"] - df["accept_time"]).dt.total_seconds() / 60
    df.loc[df["delivery_delay"] < 0, "delivery_delay"] += 24*60
    df["delivery_delay"] = df["delivery_delay"].fillna(df["delivery_delay"].median())
else:
    df["delivery_delay"] = 0

# Ensure numeric GPS
for col in ["lng","lat","accept_gps_lng","accept_gps_lat","delivery_gps_lng","delivery_gps_lat"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# GPS mismatch
df["gps_mismatch"] = df.apply(
    lambda row: haversine_km(
        row.get("accept_gps_lng",0), row.get("accept_gps_lat",0),
        row.get("delivery_gps_lng",0), row.get("delivery_gps_lat",0)
    ), axis=1
)

# Reported vs actual distance
df["reported_vs_actual_distance"] = df.apply(
    lambda row: haversine_km(
        row.get("lng",0), row.get("lat",0),
        row.get("delivery_gps_lng",0), row.get("delivery_gps_lat",0)
    ), axis=1
)

# Prepare features
feature_cols = ["delivery_delay", "gps_mismatch", "reported_vs_actual_distance"]
X = df[feature_cols].fillna(0)

# Drop any fully empty columns
X = X.loc[:, X.notna().any()]

# Fit Isolation Forest (auto contamination)
iso_forest = IsolationForest(
    contamination="auto",
    random_state=RANDOM_SEED
)
iso_forest.fit(X)

# Compute anomaly scores
df["fraud_score"] = -iso_forest.decision_function(X)  # higher = more anomalous

# Dynamic threshold: top 5% as fraud
threshold = np.percentile(df["fraud_score"], PERCENTILE_THRESHOLD)
df["fraud_label"] = (df["fraud_score"] > threshold).astype(int)  # 1 = fraud, 0 = normal

# Save predictions
df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")

# Overall statistics
total_rows = len(df)
fraud_rows = df["fraud_label"].sum()
print(f"Total rows: {total_rows}")
print(f"Detected anomalies (fraud): {fraud_rows}")
print(f"Fraction detected as fraud: {fraud_rows/total_rows:.4f}")

print("\nFraud score summary:")
print(df["fraud_score"].describe())

# Plot fraud score distribution
plt.figure(figsize=(8,5))
plt.hist(df["fraud_score"], bins=50, color='skyblue', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold ({PERCENTILE_THRESHOLD}th)")
plt.title("Distribution of Fraud Scores")
plt.xlabel("Fraud Score (higher = more likely fraud)")
plt.ylabel("Count")
plt.legend()
plt.savefig("fraud_score_distribution.png")
