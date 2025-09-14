from datasets import load_dataset
import pandas as pd
import os

# Base cache path (Hugging Face will store datasets here)
hf_cache_path = os.path.expanduser("~/.cache/huggingface/hub/datasets--Cainiao-AI--LaDe/snapshots/be2cec02775cafc8d52230303f32134382bcc50b")

# -------------------------------
# 1️⃣ Load Delivery datasets
# -------------------------------
delivery_splits = {
    "CQ": "delivery/delivery_cq.csv",
    "HZ": "delivery/delivery_hz.csv",
    "SH": "delivery/delivery_sh.csv",
    "YT": "delivery/delivery_yt.csv",
    "Five-Cities": "delivery/delivery_five_cities.csv"
}

delivery_dfs = {}
for city, path in delivery_splits.items():
    print(f"Downloading Delivery {city}...")
    ds = load_dataset("Cainiao-AI/LaDe", data_files=path, split="train")
    delivery_dfs[city] = pd.DataFrame(ds)
    print(f"✅ Delivery {city} loaded: {delivery_dfs[city].shape[0]} rows")

# -------------------------------
# 2️⃣ Load Pickup datasets
# -------------------------------
pickup_splits = {
    "CQ": "pickup/pickup_cq.csv",
    "HZ": "pickup/pickup_hz.csv",
    "SH": "pickup/pickup_sh.csv",
    "YT": "pickup/pickup_yt.csv",
    "Five-Cities": "pickup/pickup_five_cities.csv"
}

pickup_dfs = {}
for city, path in pickup_splits.items():
    print(f"Downloading Pickup {city}...")
    ds = load_dataset("Cainiao-AI/LaDe", data_files=path, split="train")
    pickup_dfs[city] = pd.DataFrame(ds)
    print(f"✅ Pickup {city} loaded: {pickup_dfs[city].shape[0]} rows")

# -------------------------------
# 3️⃣ Load Roads dataset
# -------------------------------
road_path = os.path.join(hf_cache_path, "road-network/roads.csv")

# Column names in roads.csv
road_cols = ["osm_id", "code", "fclass", "name", "ref", "oneway", 
             "maxspeed", "layer", "bridge", "tunnel", "city", "geometry"]

# Read roads.csv safely
roads_df = pd.read_csv(
    road_path,
    sep=r'\s+',      # split by spaces (multiple spaces)
    engine='python', # safer parser for irregular spacing
    names=road_cols,
    header=0,
    index_col=False
)

print(f"✅ Roads loaded: {roads_df.shape[0]} rows and {roads_df.shape[1]} columns")
print(roads_df.head())

# -------------------------------
# 4️⃣ Optional: Save datasets locally
# -------------------------------
os.makedirs("data/processed", exist_ok=True)

for city, df in delivery_dfs.items():
    df.to_csv(f"data/processed/delivery_{city}.csv", index=False)

for city, df in pickup_dfs.items():
    df.to_csv(f"data/processed/pickup_{city}.csv", index=False)

roads_df.to_csv("data/processed/roads.csv", index=False)

print("✅ All datasets saved to data/processed/")
