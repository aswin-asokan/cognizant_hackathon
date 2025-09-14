# prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ---------------------------
# 1️⃣ Load model and scaler
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/najiya/fraud shipment identification/models/gru_model.pth"
scaler_path = "/home/najiya/fraud shipment identification/models/scaler.pkl"

# Define GRU model architecture
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

# Load trained model and scaler
model = GRUModel(input_dim=7, hidden_dim=64).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
scaler = joblib.load(scaler_path)

# ---------------------------
# 2️⃣ Streamlit UI
# ---------------------------
st.title("Fraud Shipment Prediction")
st.write("Upload a CSV file to predict if a shipment is fraudulent or not.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_features = ["lng","lat","accept_gps_lng","accept_gps_lat",
                         "delivery_gps_lng","delivery_gps_lat","delivery_delay"]
    
    if not all(feat in df.columns for feat in required_features):
        st.error(f"CSV must contain columns: {required_features}")
    else:
        X = df[required_features].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            y_prob = model(X_tensor).squeeze().cpu().numpy()

        # 🔹 Set dynamic threshold for Fraud vs Not Fraud
        threshold = 0.95  # You can adjust this based on your risk preference
        y_pred = (y_prob > threshold).astype(int)

        # Create output dataframe
        df["Fraud Probability"] = y_prob
        df["Fraud Prediction"] = np.where(y_pred==1, "Fraud", "Not Fraud")
        df["Fraud Type"] = np.where(y_pred==1, "Delivery Delay", "-")
        df["Fraud Location"] = np.where(y_pred==1, df["lng"].astype(str) + "," + df["lat"].astype(str), "-")

        st.subheader("Prediction Results")
        st.write(df)

        # Allow download of results
        df.to_csv("prediction_results.csv", index=False)
        st.success("Prediction completed! You can download the results as CSV.")
