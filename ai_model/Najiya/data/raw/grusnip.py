import pandas as pd

file_path = "/home/najiya/fraud shipment identification/data/processed/combined_shipments.parquet"
df = pd.read_parquet(file_path)

print("Columns in dataset:", df.columns.tolist())
print("\nSample rows:\n", df.head())
print("\nColumn data types:\n", df.dtypes)
