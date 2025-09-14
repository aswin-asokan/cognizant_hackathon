import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.ensemble import IsolationForest

INPUT_CSV = "test_shipments.csv"     # Input CSV
OUTPUT_CSV = "predicted_fraud_if.csv"
DUMMY_YEAR = "2025"
RANDOM_SEED = 42
CONTAMINATION = 0.2  # fraction of expected fraud

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c

df = pd.read_csv(INPUT_CSV)

for col in ["accept_time", "delivery_time"]:
    if col in df.columns:
        df[col] = pd.to_datetime(
            DUMMY_YEAR + "-" + df[col].astype(str).str.strip(),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )

# Delivery delay in minutes
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

feature_cols = ["delivery_delay", "gps_mismatch", "reported_vs_actual_distance"]
X = df[feature_cols].fillna(0)

# Drop columns that are entirely empty (optional)
X = X.loc[:, X.notna().any()]

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_SEED
)
iso_forest.fit(X)

# Predict fraud
df["fraud_label"] = iso_forest.predict(X)  # -1 = anomaly, 1 = normal
df["fraud_label"] = df["fraud_label"].map({1:0, -1:1})  # convert to 0 = normal, 1 = fraud

# Optional: anomaly score
df["fraud_score"] = iso_forest.decision_function(X)  # lower = more anomalous

df.to_csv(OUTPUT_CSV, index=False)
print(f"Fraud detection completed. Predictions saved to {OUTPUT_CSV}")
