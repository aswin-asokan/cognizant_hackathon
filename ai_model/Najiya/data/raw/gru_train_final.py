# gru_train_final_memory_efficient.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# 1️⃣ Load labeled dataset
# ---------------------------
df = pd.read_parquet("/home/najiya/fraud shipment identification/data/processed/labeled_shipments.parquet")

if "label" not in df.columns:
    raise ValueError("❌ No 'label' column found in the dataset.")

features = ["lng", "lat", "accept_gps_lng", "accept_gps_lat",
            "delivery_gps_lng", "delivery_gps_lat", "delivery_delay"]

X = df[features].values
y = df["label"].values

print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print("Label distribution before oversampling:", Counter(y))

# ---------------------------
# 2️⃣ Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 3️⃣ Oversample minority class
# ---------------------------
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Take only 1 million samples to fit GPU memory
if X_train_res.shape[0] > 1_000_000:
    idx = np.random.choice(X_train_res.shape[0], 1_000_000, replace=False)
    X_train_res = X_train_res[idx]
    y_train_res = y_train_res[idx]

print("Label distribution after oversampling & subsetting:", Counter(y_train_res))

# ---------------------------
# 4️⃣ Normalize features
# ---------------------------
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# ---------------------------
# 5️⃣ Convert to tensors and create DataLoader
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_res, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# ---------------------------
# 6️⃣ Define GRU model
# ---------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

model = GRUModel(input_dim=len(features), hidden_dim=64).to(device)

# ---------------------------
# 7️⃣ Weighted BCE loss
# ---------------------------
class_counts = Counter(y_train_res)
weight_0 = class_counts[1] / class_counts[0]
weight_1 = 1.0

# Use per-sample weights in DataLoader
def get_weights(labels):
    return torch.tensor([weight_0 if l==0 else weight_1 for l in labels], dtype=torch.float32).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 8️⃣ Training loop with mini-batches
# ---------------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_weights = get_weights(batch_y)

        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = nn.BCELoss(weight=batch_weights)(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(train_loader):.4f}")
# ---------------------------
# 9️⃣ Evaluation in batches
# ---------------------------
model.eval()
y_pred_prob_list = []

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        batch_pred = model(batch_X).squeeze().cpu().numpy()
        y_pred_prob_list.append(batch_pred)

# Concatenate all batch predictions
y_pred_prob = np.concatenate(y_pred_prob_list)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n📊 Final Evaluation Metrics:")
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred, digits=4))
print("✅ ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
import os
import joblib
import torch

# Create folder if it doesn't exist
save_dir = "/home/najiya/fraud shipment identification/models"
os.makedirs(save_dir, exist_ok=True)

# Save the trained GRU model
model_path = os.path.join(save_dir, "gru_model.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Trained model saved at: {model_path}")

# Save the scaler
scaler_path = os.path.join(save_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved at: {scaler_path}")
