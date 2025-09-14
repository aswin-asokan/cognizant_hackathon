#!/usr/bin/env python3
"""
Safe training script for fraud detection.
Handles missing data, explicit datetime parsing, cross-midnight deliveries,
and encodes categorical features. Trains XGBoost and saves model + scaler + encoders.
"""

import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# --------------------------
# CONFIG
# --------------------------
INPUT_CSV = "lade_fraud_60_40_features.csv"
MODEL_OUT = "fraud_detection_xgb.pkl"
SCALER_OUT = "scaler.pkl"
ENCODERS_OUT = "label_encoders.joblib"
DUMMY_YEAR = "2025"

# --------------------------
# Haversine distance
# --------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    """Return distance in km between two GPS points"""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv(INPUT_CSV)
print("Initial rows:", len(df))

# --------------------------
# Parse datetime columns
# --------------------------
for col in ["accept_time", "delivery_time"]:
    if col in df.columns:
        df[col] = pd.to_datetime(
            DUMMY_YEAR + "-" + df[col].astype(str).str.strip(),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )

# --------------------------
# Delivery delay (minutes) and cross-midnight fix
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
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------
# Engineered GPS distances
# --------------------------
def safe_haversine(row):
    try:
        return haversine_km(
            row["accept_gps_lng"], row["accept_gps_lat"],
            row["delivery_gps_lng"], row["delivery_gps_lat"]
        )
    except:
        return 0

df["gps_mismatch"] = df.apply(safe_haversine, axis=1)

def safe_reported_vs_actual(row):
    try:
        return haversine_km(
            row["lng"], row["lat"],
            row["delivery_gps_lng"], row["delivery_gps_lat"]
        )
    except:
        return 0

df["reported_vs_actual_distance"] = df.apply(safe_reported_vs_actual, axis=1)

# --------------------------
# Fill remaining NaNs with 0
# --------------------------
df = df.fillna(0)

# --------------------------
# Categorical encoding
# --------------------------
categorical_cols = [c for c in ["region_id","city","aoi_type","courier_id"] if c in df.columns]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col+"_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

cat_feature_cols = [c+"_enc" for c in categorical_cols]
num_feature_cols = ["delivery_delay","gps_mismatch","reported_vs_actual_distance"]
feature_cols = [c for c in num_feature_cols + cat_feature_cols if c in df.columns]

X = df[feature_cols].copy()
y = df["fraud"].astype(int)

print("\nFeatures used:", feature_cols)
print("Class distribution:\n", y.value_counts())

# --------------------------
# Scale numeric features
# --------------------------
scaler = StandardScaler()
X[num_feature_cols] = scaler.fit_transform(X[num_feature_cols])

# --------------------------
# Train/test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# --------------------------
# XGBoost classifier
# --------------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1
)
model.fit(X_train, y_train)

# --------------------------
# Evaluation
# --------------------------
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# Save model, scaler, encoders
# --------------------------
joblib.dump(model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
joblib.dump(encoders, ENCODERS_OUT)
print(f"\nSaved: {MODEL_OUT}, {SCALER_OUT}, {ENCODERS_OUT}")
