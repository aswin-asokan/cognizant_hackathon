# Fraud Shipment Detection Model - Complete Pipeline
# Using Random Forest for Anomaly Detection/Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=== FRAUD SHIPMENT DETECTION MODEL ===")
print("Starting the complete pipeline...\n")

# ========================================
# 1. DATA LOADING AND INITIAL INSPECTION
# ========================================

def load_and_inspect_data(file_path):
    """Load data and perform initial inspection"""
    print("1. LOADING AND INSPECTING DATA")
    print("-" * 40)
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    if 'label' in df.columns:
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        print(f"Fraud percentage: {(df['label'].sum() / len(df)) * 100:.2f}%")
    
    return df

# ========================================
# 2. DATA PREPROCESSING AND FEATURE ENGINEERING
# ========================================

def preprocess_data(df):
    """Comprehensive data preprocessing"""
    print("\n2. DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("-" * 50)
    
    df_processed = df.copy()
    
    # Handle datetime columns (fix the hashtag issue in Excel)
    datetime_columns = ['pickup_time', 'delivery_time']
    
    for col in datetime_columns:
        if col in df_processed.columns:
            print(f"Processing {col}...")
            try:
                # Convert to datetime, handling various formats
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                print(f"  - Successfully converted {col} to datetime")
            except:
                print(f"  - Warning: Could not convert {col} to datetime")
    
    # Feature Engineering
    print("\nCreating new features...")
    
    # 1. Transit time (delivery - pickup)
    if 'pickup_time' in df_processed.columns and 'delivery_time' in df_processed.columns:
        df_processed['transit_time_hours'] = (
            df_processed['delivery_time'] - df_processed['pickup_time']
        ).dt.total_seconds() / 3600
        print("  - Created transit_time_hours")
    
    # 2. Distance calculation (Haversine formula)
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
    
    # Calculate direct distance
    df_processed['direct_distance_km'] = haversine_distance(
        df_processed['pickup_latitude'],
        df_processed['pickup_longitude'],
        df_processed['delivery_latitude'],
        df_processed['delivery_longitude']
    )
    print("  - Created direct_distance_km")
    
    # 3. Route efficiency (deviation vs direct distance)
    df_processed['route_efficiency'] = df_processed['direct_distance_km'] / (
        df_processed['direct_distance_km'] + df_processed['route_deviation_km'] + 0.001
    )
    print("  - Created route_efficiency")
    
    # 4. Speed calculation
    df_processed['average_speed_kmh'] = df_processed['direct_distance_km'] / (
        df_processed['transit_time_hours'] + 0.001
    )
    print("  - Created average_speed_kmh")
    
    # 5. Time-based features
    if 'pickup_time' in df_processed.columns:
        df_processed['pickup_hour'] = df_processed['pickup_time'].dt.hour
        df_processed['pickup_day_of_week'] = df_processed['pickup_time'].dt.dayofweek
        df_processed['pickup_month'] = df_processed['pickup_time'].dt.month
        print("  - Created pickup time features")
    
    # 6. Location-based features (zones/regions)
    df_processed['pickup_lat_rounded'] = df_processed['pickup_latitude'].round(1)
    df_processed['pickup_lon_rounded'] = df_processed['pickup_longitude'].round(1)
    df_processed['delivery_lat_rounded'] = df_processed['delivery_latitude'].round(1)
    df_processed['delivery_lon_rounded'] = df_processed['delivery_longitude'].round(1)
    print("  - Created location zone features")
    
    # Handle outliers and anomalies
    print("\nHandling outliers...")
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'label']
    
    for col in numeric_columns:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them (important for fraud detection)
        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
    
    print("  - Outliers capped using IQR method")
    
    return df_processed

# ========================================
# 3. EXPLORATORY DATA ANALYSIS
# ========================================

def perform_eda(df):
    """Perform exploratory data analysis"""
    print("\n3. EXPLORATORY DATA ANALYSIS")
    print("-" * 35)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Label distribution
    if 'label' in df.columns:
        df['label'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Fraud vs Normal Shipments')
        axes[0,0].set_xlabel('Label (0=Normal, 1=Fraud)')
        axes[0,0].set_ylabel('Count')
    
    # 2. Route deviation distribution
    if 'route_deviation_km' in df.columns:
        axes[0,1].hist(df['route_deviation_km'], bins=50, alpha=0.7)
        axes[0,1].set_title('Route Deviation Distribution')
        axes[0,1].set_xlabel('Route Deviation (km)')
        axes[0,1].set_ylabel('Frequency')
    
    # 3. Transit time distribution
    if 'transit_time_hours' in df.columns:
        axes[0,2].hist(df['transit_time_hours'], bins=50, alpha=0.7)
        axes[0,2].set_title('Transit Time Distribution')
        axes[0,2].set_xlabel('Transit Time (hours)')
        axes[0,2].set_ylabel('Frequency')
    
    # 4. Distance vs Route Deviation
    if 'direct_distance_km' in df.columns and 'route_deviation_km' in df.columns:
        scatter = axes[1,0].scatter(df['direct_distance_km'], df['route_deviation_km'], 
                                  c=df['label'] if 'label' in df.columns else 'blue', 
                                  alpha=0.6)
        axes[1,0].set_title('Distance vs Route Deviation')
        axes[1,0].set_xlabel('Direct Distance (km)')
        axes[1,0].set_ylabel('Route Deviation (km)')
        if 'label' in df.columns:
            plt.colorbar(scatter, ax=axes[1,0])
    
    # 5. Average speed distribution
    if 'average_speed_kmh' in df.columns:
        axes[1,1].hist(df['average_speed_kmh'], bins=50, alpha=0.7)
        axes[1,1].set_title('Average Speed Distribution')
        axes[1,1].set_xlabel('Average Speed (km/h)')
        axes[1,1].set_ylabel('Frequency')
    
    # 6. Pickup hour patterns
    if 'pickup_hour' in df.columns:
        df['pickup_hour'].value_counts().sort_index().plot(kind='bar', ax=axes[1,2])
        axes[1,2].set_title('Pickup Hour Distribution')
        axes[1,2].set_xlabel('Hour of Day')
        axes[1,2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis
    if 'label' in df.columns:
        print("\nCorrelation with fraud label:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['label'].abs().sort_values(ascending=False)
        print(correlations.head(10))

# ========================================
# 4. FEATURE PREPARATION
# ========================================

def prepare_features(df):
    """Prepare features for modeling"""
    print("\n4. FEATURE PREPARATION")
    print("-" * 25)
    
    # Select features for modeling
    feature_columns = [
        'pickup_latitude', 'pickup_longitude', 'delivery_latitude', 'delivery_longitude',
        'route_deviation_km', 'transit_time_hours', 'direct_distance_km',
        'route_efficiency', 'average_speed_kmh', 'pickup_hour', 'pickup_day_of_week',
        'pickup_month', 'pickup_lat_rounded', 'pickup_lon_rounded',
        'delivery_lat_rounded', 'delivery_lon_rounded'
    ]
    
    # Filter existing columns
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    X = df[available_features].copy()
    y = df['label'].copy() if 'label' in df.columns else None
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    print(f"Feature matrix shape: {X.shape}")
    if y is not None:
        print(f"Target vector shape: {y.shape}")
    
    return X, y

# ========================================
# 5. MODEL TRAINING AND EVALUATION
# ========================================

def train_random_forest(X, y):
    """Train Random Forest model with hyperparameter tuning"""
    print("\n5. MODEL TRAINING AND EVALUATION")
    print("-" * 35)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nFeatures scaled using StandardScaler")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }
    
    # Use a smaller grid for faster execution (especially for Colab CPU)
    small_param_grid = {
        'n_estimators': [50, 100],  # Reduced from [100, 200]
        'max_depth': [10, 15],      # Reduced from [10, 20] 
        'min_samples_split': [5],   # Reduced from [2, 5]
        'min_samples_leaf': [2],    # Reduced from [1, 2]
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, small_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Train final model
    best_rf = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_rf.predict(X_test_scaled)
    y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
    
    return best_rf, scaler, X_test, y_test, y_pred, y_pred_proba

# ========================================
# 6. MODEL EVALUATION AND VISUALIZATION
# ========================================

def evaluate_model(y_test, y_pred, y_pred_proba, feature_names, model):
    """Comprehensive model evaluation"""
    print("\n6. MODEL EVALUATION")
    print("-" * 20)
    
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")
    
    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
    
    axes[2].bar(range(len(indices)), importances[indices])
    axes[2].set_title('Top 10 Feature Importances')
    axes[2].set_xlabel('Features')
    axes[2].set_ylabel('Importance')
    axes[2].set_xticks(range(len(indices)))
    axes[2].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Most Important Features:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# ========================================
# 7. PREDICTION FUNCTION
# ========================================

def predict_fraud(model, scaler, new_data):
    """Function to predict fraud on new data"""
    print("\n7. PREDICTION ON NEW DATA")
    print("-" * 27)
    
    # Scale new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)[:, 1]
    
    results = pd.DataFrame({
        'Prediction': predictions,
        'Fraud_Probability': probabilities,
        'Risk_Level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in probabilities]
    })
    
    return results

# ========================================
# MAIN EXECUTION PIPELINE
# ========================================

def main(file_path='synthetic_fraud_shipments.csv'):
    """Main execution pipeline"""
    try:
        # 1. Load and inspect data
        df = load_and_inspect_data(file_path)
        
        # 2. Preprocess data
        df_processed = preprocess_data(df)
        
        # 3. EDA
        perform_eda(df_processed)
        
        # 4. Prepare features
        X, y = prepare_features(df_processed)
        
        if y is None:
            print("No label column found. Cannot train supervised model.")
            return
        
        # 5. Train model
        model, scaler, X_test, y_test, y_pred, y_pred_proba = train_random_forest(X, y)
        
        # 6. Evaluate model
        evaluate_model(y_test, y_pred, y_pred_proba, X.columns.tolist(), model)
        
        # 7. Save model components
        import joblib
        joblib.dump(model, 'fraud_detection_model.pkl')
        joblib.dump(scaler, 'fraud_detection_scaler.pkl')
        
        print("\n=== MODEL TRAINING COMPLETED ===")
        print("Model and scaler saved to disk.")
        print("- fraud_detection_model.pkl")
        print("- fraud_detection_scaler.pkl")
        
        return model, scaler, df_processed
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Run the complete pipeline
if __name__ == "__main__":
    # Execute the main pipeline
    model, scaler, processed_data = main()
    
    # Example of how to use the trained model for new predictions
    print("\n=== EXAMPLE PREDICTION ===")
    print("To predict fraud on new data:")
    print("predictions = predict_fraud(model, scaler, new_shipment_data)")
    