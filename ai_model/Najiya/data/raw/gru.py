import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1️⃣ Load dataset
input_file = "/home/najiya/fraud shipment identification/data/processed/combined_shipments.parquet"
df = pd.read_parquet(input_file)

# Make sure 'label' column exists (0=normal, 1=fraud)
if 'label' not in df.columns:
    raise ValueError("Dataset has no 'label' column. Check your LaDe dataset version!")

# 2️⃣ Select numeric features
features = df.select_dtypes(include=[np.number]).drop(columns=['label']).columns.tolist()
X = df[features].values
y = df['label'].values

# 3️⃣ Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to tensors & reshape for GRU (batch, seq_len, features)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 5️⃣ Define GRU Model
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)

model = GRUNet(input_size=X_train.shape[2], hidden_size=32, num_layers=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6️⃣ Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train).squeeze()
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 7️⃣ Evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test).squeeze()
    y_pred_labels = (y_pred_test >= 0.5).int()

# Compute metrics
cm = confusion_matrix(y_test, y_pred_labels)
report = classification_report(y_test, y_pred_labels, digits=4)
accuracy = accuracy_score(y_test, y_pred_labels)

print("\n✅ Confusion Matrix:")
print(cm)
print("\n📊 Classification Report:")
print(report)
print(f"🔎 Accuracy: {accuracy:.4f}")
