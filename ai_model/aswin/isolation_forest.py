import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# --------------------------
# Haversine distance function
# --------------------------
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c  # distance in km

# --------------------------
# Load dataset
# --------------------------
dataset_dict = load_dataset("Cainiao-AI/LaDe-D")
dfs = [split_data.to_pandas() for _, split_data in dataset_dict.items()]
df = pd.concat(dfs, ignore_index=True)
print("Total rows:", len(df))

# --------------------------
# Preprocessing
# --------------------------
df = df.dropna()

# Convert time columns to datetime
# Convert time columns to datetime with known format (MM-DD HH:MM:SS)
time_cols = ["accept_time", "delivery_time"]

for col in time_cols:
    df[col] = pd.to_datetime(
        "2025-" + df[col].astype(str),  # prepend a year
        format="%Y-%m-%d %H:%M:%S",     # parse correctly
        errors="coerce"
    )

# Drop rows where time conversion failed
df = df.dropna(subset=time_cols)

# Feature 1: delivery delay (seconds)
df["delivery_delay"] = (df["delivery_time"] - df["accept_time"]).dt.total_seconds()

# Feature 2: GPS mismatch (accept vs delivery)
df["gps_mismatch"] = df.apply(
    lambda row: haversine(row["accept_gps_lng"], row["accept_gps_lat"],
                          row["delivery_gps_lng"], row["delivery_gps_lat"]),
    axis=1
)

# Feature 3: Reported vs actual distance (lng/lat vs delivery_gps)
df["reported_vs_actual_distance"] = df.apply(
    lambda row: haversine(row["lng"], row["lat"],
                          row["delivery_gps_lng"], row["delivery_gps_lat"]),
    axis=1
)

# Encode categorical features
categorical_cols = ["city", "aoi_type"]
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# --------------------------
# Select features for anomaly detection
# --------------------------
features = [
    "region_id", "courier_id", "lng", "lat",
    "aoi_id", "city", "aoi_type",
    "accept_gps_lng", "accept_gps_lat",
    "delivery_gps_lng", "delivery_gps_lat",
    "delivery_delay", "gps_mismatch", "reported_vs_actual_distance"
]

X = df[features].values

# --------------------------
# Train Isolation Forest
# --------------------------
print("Training IsolationForest with engineered features...")
iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

df["fraud"] = iso.fit_predict(X)
df["fraud"] = df["fraud"].map({-1: 1, 1: 0})

# --------------------------
# Balanced 60-40 sample
# --------------------------
total_rows = 500000
fraud_target = int(total_rows * 0.6)      # 300k fraud
nonfraud_target = total_rows - fraud_target  # 200k non-fraud

fraud_df = df[df["fraud"] == 1]
nonfraud_df = df[df["fraud"] == 0]

if len(fraud_df) < fraud_target:
    fraud_sampled = fraud_df.sample(n=fraud_target, replace=True, random_state=42)
else:
    fraud_sampled = fraud_df.sample(n=fraud_target, random_state=42)

nonfraud_sampled = nonfraud_df.sample(n=nonfraud_target, random_state=42)

df_balanced = pd.concat([fraud_sampled, nonfraud_sampled], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42)  # shuffle

# --------------------------
# Save to CSV
# --------------------------
output_file = "lade_fraud_60_40_features.csv"
df_balanced.to_csv(output_file, index=False)

print(f"Saved {len(df_balanced)} rows to {output_file}")
print("Fraud count:", df_balanced['fraud'].sum())
print("Non-fraud count:", len(df_balanced) - df_balanced['fraud'].sum())
