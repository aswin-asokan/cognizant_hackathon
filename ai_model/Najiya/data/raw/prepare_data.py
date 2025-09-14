import pandas as pd
import os
from datasets import Dataset

# 1️⃣ Define paths (added delivery_yt)
files = {
    "delivery_sh": "/home/najiya/fraud shipment identification/base line models/delivery_sh/data-00000-of-00001.arrow",
    "delivery_jl": "/home/najiya/fraud shipment identification/base line models/delivery_jl/data-00000-of-00001.arrow",
    "delivery_hz": "/home/najiya/fraud shipment identification/base line models/delivery_hz/data-00000-of-00001.arrow",
    "delivery_cq": "/home/najiya/fraud shipment identification/base line models/delivery_cq/data-00000-of-00001.arrow",
    "delivery_yt": "/home/najiya/fraud shipment identification/base line models/delivery_yt/data-00000-of-00001.arrow",
}

# 2️⃣ Load all datasets
dfs = []
for name, path in files.items():
    print(f"Loading {name}...")
    ds = Dataset.from_file(path)
    df = ds.to_pandas()
    df["source"] = name  # track which city/region the record came from
    dfs.append(df)

# 3️⃣ Combine into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
print(f"✅ Combined dataset shape: {combined_df.shape}")

# 4️⃣ Handle missing values
for col in combined_df.columns:
    if combined_df[col].dtype in ["int64", "float64"]:
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())
    else:
        combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

print("✅ Missing values handled.")

# 5️⃣ Ensure processed folder exists
output_folder = "/home/najiya/fraud shipment identification/data/processed"
os.makedirs(output_folder, exist_ok=True)

# 6️⃣ Save combined data
output_file = os.path.join(output_folder, "combined_shipments.parquet")
combined_df.to_parquet(output_file, index=False)
print(f"💾 Saved cleaned dataset to {output_file}")
