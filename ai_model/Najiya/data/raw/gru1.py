# gru_train_final.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter

# 1️⃣ Load labeled dataset
df = pd.read_parquet("/home/najiya/fraud shipment identification/data/processed/labeled_shipments.parquet")

# Ensure label column exists
if "label" not in df.columns:
    raise ValueError("❌ No 'label' column found in the dataset. Cannot train supervised model.")

# 2️⃣ Select numeric features + target
features = ["lng", "lat", "accept_gps_lng", "accept_gps_lat", 
            "delivery_gps_lng", "delivery_gps_lat", "delivery_delay"]
X = df[features].values
y = df["label"].values

print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print("Label distribution:", Counter(y))

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 6️⃣ Define GRU model
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

model = GRUModel(input_dim=len(features), hidden_dim=64)

# Use weighted loss for class imbalance
class_weights = torch.tensor([len(y) / (2 * np.sum(y == c)) for c in [0, 1]], dtype=torch.float32)
criterion = nn.BCELoss(weight=(y_train_tensor * class_weights[1] + (1 - y_train_tensor) * class_weights[0]))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7️⃣ Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor).squeeze()
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

# 8️⃣ Evaluation
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).squeeze().numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)

print("\n📊 Final Evaluation Metrics:")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred, digits=4))
