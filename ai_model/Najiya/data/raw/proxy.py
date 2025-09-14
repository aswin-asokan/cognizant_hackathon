# generate_labels.py
import pandas as pd
import numpy as np

input_file = "/home/najiya/fraud shipment identification/data/processed/combined_shipments.parquet"
df = pd.read_parquet(input_file)

# Convert times to datetime
df["accept_time"] = pd.to_datetime(df["accept_time"], errors="coerce")
df["delivery_time"] = pd.to_datetime(df["delivery_time"], errors="coerce")

# Calculate delivery delay in minutes
df["delivery_delay"] = (df["delivery_time"] - df["accept_time"]).dt.total_seconds() / 60

# Drop rows where delay could not be calculated
df = df.dropna(subset=["delivery_delay"])

# Calculate threshold (95th percentile)
threshold = df["delivery_delay"].quantile(0.95)
print(f"95th percentile threshold: {threshold:.2f} minutes")

# Assign labels
df["label"] = np.where(df["delivery_delay"] > threshold, 1, 0)

print(df["label"].value_counts(normalize=True))

# Save labeled dataset
output_file = "/home/najiya/fraud shipment identification/data/processed/labeled_shipments.parquet"
df.to_parquet(output_file, index=False)
print(f"✅ Labeled dataset saved to {output_file}")
