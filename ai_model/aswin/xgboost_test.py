import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2

# --------------------------
# Config
# --------------------------
INPUT_CSV = "test_shipments.csv"          # Your new CSV
OUTPUT_CSV = "predicted_fraud.csv"
MODEL_OUT = "fraud_detection_xgb.pkl"
SCALER_OUT = "scaler.pkl"
ENCODERS_OUT = "label_encoders.joblib"
DUMMY_YEAR = "2025"

# --------------------------
# Haversine distance
# --------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# --------------------------
# Load model, scaler, encoders
# --------------------------
model = joblib.load(MODEL_OUT)
scaler = joblib.load(SCALER_OUT)
encoders = joblib.load(ENCODERS_OUT)

# --------------------------
# Load new data
# --------------------------
df = pd.read_csv(INPUT_CSV)

# --------------------------
# Parse datetime safely
# --------------------------
for col in ["accept_time", "delivery_time"]:
    if col in df.columns:
        df[col] = pd.to_datetime(
            DUMMY_YEAR + "-" + df[col].astype(str).str.strip(),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )

# --------------------------
# Compute delivery delay in minutes
# --------------------------
if set(["accept_time", "delivery_time"]).issubset(df.columns):
    df["delivery_delay"] = (df["delivery_time"] - df["accept_time"]).dt.total_seconds() / 60
    df.loc[df["delivery_delay"] < 0, "delivery_delay"] += 24*60
    df["delivery_delay"] = df["delivery_delay"].fillna(df["delivery_delay"].median())
else:
    df["delivery_delay"] = 0

# --------------------------
# Ensure numeric GPS
# --------------------------
for col in ["lng","lat","accept_gps_lng","accept_gps_lat","delivery_gps_lng","delivery_gps_lat"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# --------------------------
# Compute GPS distances
# --------------------------
df["gps_mismatch"] = df.apply(
    lambda row: haversine_km(
        row.get("accept_gps_lng", 0), row.get("accept_gps_lat", 0),
        row.get("delivery_gps_lng", 0), row.get("delivery_gps_lat", 0)
    ), axis=1
)

df["reported_vs_actual_distance"] = df.apply(
    lambda row: haversine_km(
        row.get("lng", 0), row.get("lat", 0),
        row.get("delivery_gps_lng", 0), row.get("delivery_gps_lat", 0)
    ), axis=1
)

# --------------------------
# Encode categorical features safely
# --------------------------
for col, le in encoders.items():
    if col in df.columns:
        # Map unseen categories to the most frequent class
        df[col+"_enc"] = df[col].map(
            lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else np.nan
        )
        # Fill NaN (unseen) with mode from training
        df[col+"_enc"] = df[col+"_enc"].fillna(pd.Series(le.transform([c for c in le.classes_])).mode()[0])
    else:
        df[col+"_enc"] = pd.Series([pd.Series(le.transform([c for c in le.classes_])).mode()[0]] * len(df))

# --------------------------
# Select features
# --------------------------
num_feature_cols = ["delivery_delay","gps_mismatch","reported_vs_actual_distance"]
cat_feature_cols = [c+"_enc" for c in encoders.keys()]
feature_cols = [c for c in num_feature_cols + cat_feature_cols if c in df.columns]

X = df[feature_cols].copy()

# --------------------------
# Scale numeric features
# --------------------------
X[num_feature_cols] = scaler.transform(X[num_feature_cols])

# --------------------------
# Predict fraud
# --------------------------
df["fraud_prediction"] = model.predict(X)
df["fraud_prob"] = model.predict_proba(X)[:,1]

# --------------------------
# Save results
# --------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")
