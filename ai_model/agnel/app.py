import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Shipment Detection System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .fraud-alert {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .safe-alert {
        background-color: #00aa00;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .warning-alert {
        background-color: #ff8800;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth"""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    return distance

def create_features(data):
    """Create engineered features from input data"""
    df = pd.DataFrame([data])
    
    # Calculate transit time in hours
    pickup_dt = pd.to_datetime(data['pickup_time'])
    delivery_dt = pd.to_datetime(data['delivery_time'])
    df['transit_time_hours'] = (delivery_dt - pickup_dt).total_seconds() / 3600
    
    # Calculate direct distance
    df['direct_distance_km'] = haversine_distance(
        data['pickup_latitude'], data['pickup_longitude'],
        data['delivery_latitude'], data['delivery_longitude']
    )
    
    # Calculate route efficiency
    df['route_efficiency'] = df['direct_distance_km'] / (
        df['direct_distance_km'] + data['route_deviation_km'] + 0.001
    )
    
    # Calculate average speed
    df['average_speed_kmh'] = df['direct_distance_km'] / (df['transit_time_hours'] + 0.001)
    
    # Time-based features
    df['pickup_hour'] = pickup_dt.hour
    df['pickup_day_of_week'] = pickup_dt.weekday()
    df['pickup_month'] = pickup_dt.month
    
    # Location features (rounded)
    df['pickup_lat_rounded'] = round(data['pickup_latitude'], 1)
    df['pickup_lon_rounded'] = round(data['pickup_longitude'], 1)
    df['delivery_lat_rounded'] = round(data['delivery_latitude'], 1)
    df['delivery_lon_rounded'] = round(data['delivery_longitude'], 1)
    
    # Select features for prediction (matching training features)
    feature_columns = [
        'pickup_latitude', 'pickup_longitude', 'delivery_latitude', 'delivery_longitude',
        'route_deviation_km', 'transit_time_hours', 'direct_distance_km',
        'route_efficiency', 'average_speed_kmh', 'pickup_hour', 'pickup_day_of_week',
        'pickup_month', 'pickup_lat_rounded', 'pickup_lon_rounded',
        'delivery_lat_rounded', 'delivery_lon_rounded'
    ]
    
    return df[feature_columns]

def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('fraud_detection_model.pkl')
        scaler = joblib.load('fraud_detection_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        # Create a dummy model for demonstration if files don't exist
        st.warning("Model files not found. Using dummy model for demonstration.")
        
        # Create dummy model with same structure
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        
        # Create dummy training data to fit the model
        np.random.seed(42)
        dummy_X = np.random.randn(1000, 16)
        dummy_y = np.random.choice([0, 1], 1000, p=[0.85, 0.15])
        
        scaler.fit(dummy_X)
        model.fit(scaler.transform(dummy_X), dummy_y)
        
        return model, scaler

def create_route_map(pickup_lat, pickup_lon, delivery_lat, delivery_lon):
    """Create a folium map showing the route"""
    # Center map between pickup and delivery
    center_lat = (pickup_lat + delivery_lat) / 2
    center_lon = (pickup_lon + delivery_lon) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add pickup marker
    folium.Marker(
        [pickup_lat, pickup_lon],
        popup="Pickup Location",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    # Add delivery marker
    folium.Marker(
        [delivery_lat, delivery_lon],
        popup="Delivery Location",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # Add route line
    folium.PolyLine(
        [[pickup_lat, pickup_lon], [delivery_lat, delivery_lon]],
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)
    
    return m

def main():
    # Header
    st.markdown('<div class="main-header">🚚 Fraud Shipment Detection System</div>', unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    
    # Sidebar for input method selection
    st.sidebar.title("📊 Input Method")
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Single Shipment", "Batch Upload", "Sample Data"]
    )
    
    if input_method == "Single Shipment":
        single_shipment_input(model, scaler)
    elif input_method == "Batch Upload":
        batch_upload_input(model, scaler)
    else:
        sample_data_input(model, scaler)

def single_shipment_input(model, scaler):
    """Handle single shipment input and prediction"""
    st.subheader("🔍 Single Shipment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Pickup Information")
        shipment_id = st.text_input("Shipment ID", value="SHP001")
        tracking_number = st.text_input("Tracking Number", value="TRK123456789")
        
        pickup_date = st.date_input("Pickup Date", value=datetime.now().date())
        pickup_time = st.time_input("Pickup Time", value=datetime.now().time())
        pickup_datetime = datetime.combine(pickup_date, pickup_time)
        
        pickup_lat = st.number_input("Pickup Latitude", value=40.7128, format="%.6f")
        pickup_lon = st.number_input("Pickup Longitude", value=-74.0060, format="%.6f")
    
    with col2:
        st.markdown("### 🎯 Delivery Information")
        delivery_date = st.date_input("Delivery Date", value=(datetime.now() + timedelta(days=1)).date())
        delivery_time = st.time_input("Delivery Time", value=datetime.now().time())
        delivery_datetime = datetime.combine(delivery_date, delivery_time)
        
        delivery_lat = st.number_input("Delivery Latitude", value=40.7589, format="%.6f")
        delivery_lon = st.number_input("Delivery Longitude", value=-73.9851, format="%.6f")
        
        route_deviation = st.number_input("Route Deviation (km)", value=5.0, min_value=0.0, format="%.2f")
    
    # Create map
    st.markdown("### 🗺️ Route Visualization")
    route_map = create_route_map(pickup_lat, pickup_lon, delivery_lat, delivery_lon)
    st_folium(route_map, width=700, height=400)
    
    # Prediction button
    if st.button("🔍 Analyze Shipment", type="primary"):
        # Prepare data
        shipment_data = {
            'pickup_latitude': pickup_lat,
            'pickup_longitude': pickup_lon,
            'delivery_latitude': delivery_lat,
            'delivery_longitude': delivery_lon,
            'route_deviation_km': route_deviation,
            'pickup_time': pickup_datetime,
            'delivery_time': delivery_datetime
        }
        
        # Create features
        features = create_features(shipment_data)
        
        # Make prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, 1]
        
        # Display results
        display_prediction_results(shipment_data, prediction, probability, features)

def batch_upload_input(model, scaler):
    """Handle batch file upload and prediction"""
    st.subheader("📁 Batch Shipment Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file with shipment data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded {len(df)} shipments successfully!")
            
            # Show sample data
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head())
            
            if st.button("🔍 Analyze All Shipments", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    try:
                        # Create features for each row
                        shipment_data = {
                            'pickup_latitude': row['pickup_latitude'],
                            'pickup_longitude': row['pickup_longitude'],
                            'delivery_latitude': row['delivery_latitude'],
                            'delivery_longitude': row['delivery_longitude'],
                            'route_deviation_km': row['route_deviation_km'],
                            'pickup_time': pd.to_datetime(row['pickup_time']),
                            'delivery_time': pd.to_datetime(row['delivery_time'])
                        }
                        
                        features = create_features(shipment_data)
                        features_scaled = scaler.transform(features)
                        
                        prediction = model.predict(features_scaled)[0]
                        probability = model.predict_proba(features_scaled)[0, 1]
                        
                        risk_level = "🔴 HIGH" if probability > 0.7 else "🟡 MEDIUM" if probability > 0.3 else "🟢 LOW"
                        
                        results.append({
                            'shipment_id': row.get('shipment_id', f'SHP_{idx}'),
                            'tracking_number': row.get('tracking_number', f'TRK_{idx}'),
                            'fraud_probability': probability,
                            'prediction': 'FRAUD' if prediction == 1 else 'NORMAL',
                            'risk_level': risk_level
                        })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    except Exception as e:
                        st.error(f"Error processing row {idx}: {str(e)}")
                
                # Display results
                display_batch_results(results)
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def sample_data_input(model, scaler):
    """Show sample data for testing"""
    st.subheader("🎯 Sample Data Testing")
    
    # Sample shipments with different risk profiles
    samples = {
        "Normal Delivery": {
            'pickup_latitude': 40.7128,
            'pickup_longitude': -74.0060,
            'delivery_latitude': 40.7589,
            'delivery_longitude': -73.9851,
            'route_deviation_km': 2.5,
            'pickup_time': datetime(2024, 1, 15, 9, 0),
            'delivery_time': datetime(2024, 1, 15, 11, 30)
        },
        "Suspicious Route": {
            'pickup_latitude': 40.7128,
            'pickup_longitude': -74.0060,
            'delivery_latitude': 40.7589,
            'delivery_longitude': -73.9851,
            'route_deviation_km': 25.0,  # High deviation
            'pickup_time': datetime(2024, 1, 15, 2, 0),  # Unusual time
            'delivery_time': datetime(2024, 1, 15, 2, 30)  # Too fast
        },
        "Long Distance": {
            'pickup_latitude': 40.7128,
            'pickup_longitude': -74.0060,
            'delivery_latitude': 34.0522,
            'delivery_longitude': -118.2437,
            'route_deviation_km': 15.0,
            'pickup_time': datetime(2024, 1, 15, 8, 0),
            'delivery_time': datetime(2024, 1, 17, 18, 0)
        }
    }
    
    selected_sample = st.selectbox("Choose a sample shipment:", list(samples.keys()))
    
    if st.button(f"🔍 Analyze {selected_sample}", type="primary"):
        shipment_data = samples[selected_sample]
        features = create_features(shipment_data)
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, 1]
        
        display_prediction_results(shipment_data, prediction, probability, features)

def display_prediction_results(shipment_data, prediction, probability, features):
    """Display prediction results with visualizations"""
    st.markdown("### 📊 Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.markdown('<div class="fraud-alert">⚠️ FRAUD DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-alert">✅ NORMAL SHIPMENT</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Fraud Probability", f"{probability:.1%}")
    
    with col3:
        risk_level = "🔴 HIGH" if probability > 0.7 else "🟡 MEDIUM" if probability > 0.3 else "🟢 LOW"
        st.metric("Risk Level", risk_level)
    
    # Feature analysis
    st.markdown("### 📈 Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Key metrics
        direct_dist = features['direct_distance_km'].iloc[0]
        transit_time = features['transit_time_hours'].iloc[0]
        avg_speed = features['average_speed_kmh'].iloc[0]
        route_eff = features['route_efficiency'].iloc[0]
        
        st.metric("Direct Distance", f"{direct_dist:.1f} km")
        st.metric("Transit Time", f"{transit_time:.1f} hours")
        st.metric("Average Speed", f"{avg_speed:.1f} km/h")
        st.metric("Route Efficiency", f"{route_eff:.2f}")
    
    with col2:
        # Create radar chart for risk factors
        categories = ['Route Deviation', 'Speed Anomaly', 'Time Anomaly', 'Location Risk']
        
        # Calculate risk scores (0-1)
        route_risk = min(shipment_data['route_deviation_km'] / 20.0, 1.0)
        speed_risk = min(abs(avg_speed - 50) / 50.0, 1.0)  # Assuming 50 km/h is normal
        time_risk = 1.0 if features['pickup_hour'].iloc[0] < 6 or features['pickup_hour'].iloc[0] > 22 else 0.0
        location_risk = probability  # Use model probability as location risk
        
        values = [route_risk, speed_risk, time_risk, location_risk]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Factors',
            line_color='red' if prediction == 1 else 'green'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            title="Risk Factor Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if prediction == 1:
        st.error("""
        **High Risk Shipment Detected:**
        - Conduct manual verification
        - Contact driver/carrier for explanation
        - Check customer history and payment method
        - Consider GPS tracking verification
        """)
    elif probability > 0.3:
        st.warning("""
        **Medium Risk Shipment:**
        - Monitor delivery progress closely
        - Verify delivery confirmation
        - Check for any route changes
        """)
    else:
        st.success("""
        **Low Risk Shipment:**
        - Proceed with normal processing
        - Standard monitoring sufficient
        """)

def display_batch_results(results):
    """Display batch prediction results"""
    st.markdown("### 📊 Batch Analysis Results")
    
    df_results = pd.DataFrame(results)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_shipments = len(results)
    fraud_count = sum(1 for r in results if r['prediction'] == 'FRAUD')
    high_risk = sum(1 for r in results if r['fraud_probability'] > 0.7)
    avg_risk = np.mean([r['fraud_probability'] for r in results])
    
    with col1:
        st.metric("Total Shipments", total_shipments)
    with col2:
        st.metric("Fraud Detected", fraud_count, f"{(fraud_count/total_shipments)*100:.1f}%")
    with col3:
        st.metric("High Risk", high_risk, f"{(high_risk/total_shipments)*100:.1f}%")
    with col4:
        st.metric("Average Risk", f"{avg_risk:.1%}")
    
    # Risk distribution chart
    fig = px.histogram(
        df_results, 
        x='fraud_probability', 
        nbins=20,
        title='Fraud Probability Distribution',
        color_discrete_sequence=['#FF6B6B']
    )
    fig.update_layout(xaxis_title='Fraud Probability', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.markdown("### 📋 Detailed Results")
    
    # Add color coding to dataframe
    def color_risk(val):
        if 'HIGH' in str(val):
            return 'background-color: #ffebee'
        elif 'MEDIUM' in str(val):
            return 'background-color: #fff3e0'
        else:
            return 'background-color: #e8f5e8'
    
    styled_df = df_results.style.applymap(color_risk, subset=['risk_level'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Download results
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"fraud_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
def add_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚚 Fraud Shipment Detection System | Built with Streamlit & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    add_footer()