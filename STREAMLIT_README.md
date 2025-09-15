# Streamlit Fraud Detection Frontend

A user-friendly web interface for uploading CSV files and detecting potential fraud in logistics data using the Isolation Forest algorithm.

## Features

- 📁 **CSV Upload**: Easy file upload interface
- 🔍 **Real-time Processing**: Instant fraud detection results
- 📊 **Visual Analytics**: Charts and statistics for fraud analysis
- 💾 **Download Results**: Export results as CSV files
- ⚙️ **Configurable**: Adjustable contamination rate and random seed
- 🚨 **Fraud Alerts**: Highlighted potential fraud cases

## Installation

1. Install the required dependencies:

```bash
pip install -r streamlit_requirements.txt
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run streamlit_fraud_detection.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Upload a CSV file with shipment data

4. Configure the contamination rate (default: 0.2 = 20% expected fraud rate)

5. Click "Detect Fraud" to analyze the data

6. Download the results as CSV files

## Required CSV Format

Your CSV file should contain these columns:

- `accept_time`: Time when shipment was accepted (format: HH:MM:SS)
- `delivery_time`: Time when shipment was delivered (format: HH:MM:SS)
- `lng`, `lat`: Reported longitude and latitude coordinates
- `accept_gps_lng`, `accept_gps_lat`: GPS coordinates when accepted
- `delivery_gps_lng`, `delivery_gps_lat`: GPS coordinates when delivered

## Output

The application generates:

- `fraud_label`: 1 if fraud detected, 0 if normal
- `fraud_score`: Higher scores indicate higher likelihood of fraud
- Detailed statistics and visualizations
- Separate CSV files for all results and fraud cases only

## Algorithm Details

The fraud detection uses Isolation Forest with these engineered features:

1. **Delivery Delay**: Time difference between accept and delivery
2. **GPS Mismatch**: Distance between accept and delivery GPS locations
3. **Reported vs Actual Distance**: Difference between reported coordinates and delivery GPS

