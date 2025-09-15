from datasets import load_dataset
import os

data_folder = "/home/najiya/fraud shipment identification/data/raw"

data_files = {
    "delivery_cq": os.path.join(data_folder, "delivery_cq.csv"),
    "delivery_hz": os.path.join(data_folder, "delivery_hz.csv"),
    "delivery_sh": os.path.join(data_folder, "delivery_sh.csv"),
    "delivery_yt": os.path.join(data_folder, "delivery_yt.csv"),
    "delivery_five": os.path.join(data_folder, "delivery_five.csv"),
}

ds = load_dataset("csv", data_files=data_files)
print(ds)
