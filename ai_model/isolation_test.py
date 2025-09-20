import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.ensemble import IsolationForest
from io import StringIO

RANDOM_SEED = 42
PERCENTILE_THRESHOLD = 95  # top 5% fraud

# Haversine distance
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c

def process_csv(file_stream):
    df = pd.read_csv(file_stream)

    # Dummy year for datetime parsing
    DUMMY_YEAR = "2025"
    for col in ["accept_time", "delivery_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(
                DUMMY_YEAR + "-" + df[col].astype(str).str.strip(),
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce"
            )

    # Compute delivery_delay
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

    # Features for Isolation Forest
    feature_cols = ["delivery_delay", "gps_mismatch", "reported_vs_actual_distance"]
    X = df[feature_cols].fillna(0)
    X = X.loc[:, X.notna().any()]

    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination="auto", random_state=RANDOM_SEED)
    iso_forest.fit(X)

    df["fraud_score"] = -iso_forest.decision_function(X)  # higher = more anomalous
    threshold = np.percentile(df["fraud_score"], PERCENTILE_THRESHOLD)
    df["fraud_label"] = (df["fraud_score"] > threshold).astype(int)

    normal_count = int((df["fraud_label"] == 0).sum())
    fraudulent_count = int((df["fraud_label"] == 1).sum())
    avg_score = float(df["fraud_score"].mean())

    # Return processed CSV as string
    processed_csv = df.to_csv(index=False)

    return {
        "normal_count": normal_count,
        "fraudulent_count": fraudulent_count,
        "anomaly_score": avg_score,
        "processed_csv": processed_csv
    }
