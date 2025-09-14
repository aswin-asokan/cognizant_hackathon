# weeksupervision_fixed.py
import pandas as pd
import numpy as np

# Load your combined shipments
df = pd.read_parquet("/home/najiya/fraud shipment identification/data/processed/combined_shipments.parquet")

# Combine with a dummy reference date so parsing works
def safe_parse_time(time_series):
    """Safely parse HH:MM:SS or MM-DD HH:MM:SS with a dummy reference date."""
    # Fill NaN with a placeholder
    time_series = time_series.fillna("00-00 00:00:00")

    # Attach dummy year for parsing
    time_series = "2024-" + time_series.astype(str)  # add fake year

    # Try parsing, coerce errors to NaT
    return pd.to_datetime(time_series, format="2024-%m-%d %H:%M:%S", errors="coerce")

# Parse accept and delivery times
accept_times = safe_parse_time(df["accept_time"])
delivery_times = safe_parse_time(df["delivery_time"])

# Calculate delay in seconds, safely ignoring NaT
df["delivery_delay"] = (delivery_times - accept_times).dt.total_seconds()
df["delivery_delay"] = df["delivery_delay"].fillna(0)

# Example heuristic: Mark as fraud if delay > 600 seconds or delay < 0
df["label"] = np.where((df["delivery_delay"] > 600) | (df["delivery_delay"] < 0), 1, 0)

print("Label distribution:\n", df["label"].value_counts())

# Save to parquet
output_file = "/home/najiya/fraud shipment identification/data/processed/labeled_shipments.parquet"
df.to_parquet(output_file, index=False)
print(f"✅ Labeled dataset saved to {output_file} with {df.shape[0]} rows.")
