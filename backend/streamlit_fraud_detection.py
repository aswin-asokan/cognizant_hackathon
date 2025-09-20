import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from math import radians, sin, cos, sqrt, atan2
import io

# --------------------------
# Haversine distance function
# --------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on earth in kilometers"""
    R = 6371.0  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c

# --------------------------
# Preprocessing
# --------------------------
def preprocess_data(df):
    """Preprocess the uploaded CSV data"""
    df_processed = df.copy()

    # Parse datetime columns
    for col in ["accept_time", "delivery_time"]:
        if col in df_processed.columns:
            # If we have a "ds" column and times look like HH:MM:SS → combine
            if "ds" in df_processed.columns and df_processed[col].astype(str).str.len().max() <= 8:
                df_processed[col] = pd.to_datetime(
                    df_processed["ds"].astype(str).str.strip() + " " + df_processed[col].astype(str).str.strip(),
                    errors="coerce"
                )
            else:
                # Already full datetime → parse directly
                df_processed[col] = pd.to_datetime(df_processed[col], errors="coerce")

            # Handle missing values
            if df_processed[col].isna().any():
                st.warning(f"⚠️ Some {col} values could not be parsed. Filling with median.")
                median_time = df_processed[col].dropna().median()
                df_processed[col] = df_processed[col].fillna(median_time)

    # Delivery delay (minutes)
    if set(["accept_time", "delivery_time"]).issubset(df_processed.columns):
        df_processed["delivery_delay"] = (
            (df_processed["delivery_time"] - df_processed["accept_time"]).dt.total_seconds() / 60
        )
        # Handle negative values by adding 24h
        df_processed.loc[df_processed["delivery_delay"] < 0, "delivery_delay"] += 24 * 60
        df_processed["delivery_delay"] = df_processed["delivery_delay"].fillna(
            df_processed["delivery_delay"].median()
        )
    else:
        df_processed["delivery_delay"] = 0
        st.warning("⚠️ accept_time or delivery_time columns not found. Setting delivery_delay to 0.")

    # Ensure numeric GPS
    gps_cols = ["lng", "lat", "accept_gps_lng", "accept_gps_lat", "delivery_gps_lng", "delivery_gps_lat"]
    for col in gps_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce").fillna(0)

    # GPS mismatch
    df_processed["gps_mismatch"] = df_processed.apply(
        lambda row: haversine_km(
            row.get("accept_gps_lng", 0), row.get("accept_gps_lat", 0),
            row.get("delivery_gps_lng", 0), row.get("delivery_gps_lat", 0)
        ),
        axis=1,
    )

    # Reported vs actual distance
    df_processed["reported_vs_actual_distance"] = df_processed.apply(
        lambda row: haversine_km(
            row.get("lng", 0), row.get("lat", 0),
            row.get("delivery_gps_lng", 0), row.get("delivery_gps_lat", 0)
        ),
        axis=1,
    )

    return df_processed

# --------------------------
# Fraud Detection
# --------------------------
def detect_fraud(df, random_seed=42):
    """Apply Isolation Forest for fraud detection"""
    feature_cols = ["delivery_delay", "gps_mismatch", "reported_vs_actual_distance"]
    X = df[feature_cols].fillna(0)
    X = X.loc[:, X.notna().any()]

    # Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_seed
    )
    iso_forest.fit(X)

    fraud_scores = -iso_forest.decision_function(X)  # higher = more anomalous
    return fraud_scores

# --------------------------
# Streamlit App
# --------------------------
def main():
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🛡️",
        layout="wide"
    )

    st.title("ShipGuard 🛡️")
    st.markdown("Upload a CSV file with shipment data to detect potential fraud using Isolation Forest.")

    # File upload
    st.header("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

            with st.expander("📊 Preview Data", expanded=False):
                st.dataframe(df.head(10))
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")

            # Threshold slider
            percentile = st.slider(
                "Fraud Threshold Percentile",
                min_value=80,
                max_value=99,
                value=95,
                step=1,
                help="Higher percentile → fewer rows flagged as fraud"
            )

            if st.button("🔍 Detect Fraud", type="primary"):
                with st.spinner("Processing data and detecting fraud..."):
                    df_processed = preprocess_data(df)
                    fraud_scores = detect_fraud(df_processed)
                    threshold = np.percentile(fraud_scores, percentile)

                    df_result = df_processed.copy()
                    df_result["fraud_score"] = fraud_scores
                    df_result["fraud_label"] = (fraud_scores > threshold).astype(int)

                    # Stats
                    total_rows = len(df_result)
                    fraud_rows = df_result["fraud_label"].sum()
                    fraud_rate = (fraud_rows / total_rows) * 100

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Records", total_rows)
                    col2.metric("Fraud Detected", fraud_rows)
                    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                    col4.metric("Avg Fraud Score", f"{df_result['fraud_score'].mean():.3f}")

                    # Fraud distribution
                    st.subheader("📈 Fraud Score Distribution")
                    st.bar_chart(np.histogram(fraud_scores, bins=30)[0])

                    with st.expander("📋 Detailed Results", expanded=True):
                        st.dataframe(df_result)

                    # Download results
                    csv_buffer = io.StringIO()
                    df_result.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "💾 Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )

                    fraud_cases = df_result[df_result["fraud_label"] == 1]
                    if len(fraud_cases) > 0:
                        st.subheader("🚨 Potential Fraud Cases")
                        st.dataframe(fraud_cases)

                        fraud_buffer = io.StringIO()
                        fraud_cases.to_csv(fraud_buffer, index=False)
                        st.download_button(
                            "💾 Download Fraud Cases Only",
                            data=fraud_buffer.getvalue(),
                            file_name="fraud_cases_only.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

    with st.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        ### Required CSV Format
        Columns: `accept_time`, `delivery_time`, `lng`, `lat`, 
        `accept_gps_lng`, `accept_gps_lat`, `delivery_gps_lng`, `delivery_gps_lat`
        
        ### Features Used
        - Delivery Delay
        - GPS Mismatch
        - Reported vs Actual Distance
        
        ### Output
        - fraud_label: 1 = fraud, 0 = normal
        - fraud_score: higher = more likely fraud
        """)

    st.markdown("---")
    st.markdown("*Built with Streamlit and scikit-learn Isolation Forest*")

if __name__ == "__main__":
    main()
