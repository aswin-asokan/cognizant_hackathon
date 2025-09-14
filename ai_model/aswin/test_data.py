import pandas as pd
import numpy as np

# --------------------------
# Config
# --------------------------
OUTPUT_CSV = "test_shipments.csv"
NUM_ROWS = 1000

# --------------------------
# Generate random data
# --------------------------
np.random.seed(42)

df = pd.DataFrame({
    "order_id": np.arange(1, NUM_ROWS+1),
    "region_id": np.random.choice(["R1","R2","R3","R4"], NUM_ROWS),
    "city": np.random.choice(["CityA","CityB","CityC"], NUM_ROWS),
    "courier_id": np.random.choice(["C1","C2","C3"], NUM_ROWS),
    "lng": np.random.uniform(120, 122, NUM_ROWS),
    "lat": np.random.uniform(30, 32, NUM_ROWS),
    "aoi_id": np.random.randint(1, 50, NUM_ROWS),
    "aoi_type": np.random.choice(["residential","commercial"], NUM_ROWS),
    "accept_time": np.random.randint(0, 24*60*60, NUM_ROWS),  # seconds since midnight
    "accept_gps_time": np.random.randint(0, 24*60*60, NUM_ROWS),
    "accept_gps_lng": np.random.uniform(120, 122, NUM_ROWS),
    "accept_gps_lat": np.random.uniform(30, 32, NUM_ROWS),
    "delivery_time": np.random.randint(0, 24*60*60, NUM_ROWS),
    "delivery_gps_time": np.random.randint(0, 24*60*60, NUM_ROWS),
    "delivery_gps_lng": np.random.uniform(120, 122, NUM_ROWS),
    "delivery_gps_lat": np.random.uniform(30, 32, NUM_ROWS),
    "ds": pd.Timestamp("2025-09-14")
})

# Convert times to HH:MM:SS format
def seconds_to_hms(sec):
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

df["accept_time"] = df["accept_time"].apply(seconds_to_hms)
df["accept_gps_time"] = df["accept_gps_time"].apply(seconds_to_hms)
df["delivery_time"] = df["delivery_time"].apply(seconds_to_hms)
df["delivery_gps_time"] = df["delivery_gps_time"].apply(seconds_to_hms)

# --------------------------
# Save CSV
# --------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"Test CSV generated: {OUTPUT_CSV}")
