import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import json
from datetime import datetime
from flask import Flask, request, jsonify


from flask import Flask
from flask_cors import CORS
app = Flask(_name_)
model_data = None  # will hold trained model, scaler, and features

app = Flask(_name_)
CORS(app)   # ✅ Add this here, not in train_fraud_model.py


def load_and_preprocess_data():
    """Load and preprocess the enhanced fraud dataset"""
    print("Loading enhanced fraud dataset...")
    
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/enhanced_fraud_dataset-ZMjbsPepn5xUXRFrSs12xgofaGgWk1.csv"
    df = pd.read_csv(url)
    
    # Convert numeric columns
    numeric_columns = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'avg_amount', 'transaction_count',
        'amount_std', 'amount_to_balance_ratio', 'amount_zscore', 'is_round_amount',
        'balance_drained', 'dest_transaction_count', 'is_new_dest', 'hourly_txn_count',
        'balance_change_ratio', 'unique_destinations', 'unique_senders_to_dest',
        'deviation_flag', 'high_velocity_flag', 'suspicious_time_flag',
        'txn_frequency_per_day', 'amount_risk_score', 'behavioral_risk_score', 'isFraud'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle categorical variables
    df['type_DEBIT'] = (df['type'] == 'DEBIT').astype(int)
    df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
    df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
    df['type_PAYMENT'] = (df['type'] == 'PAYMENT').astype(int)
    df['type_CASH_IN'] = (df['type'] == 'CASH_IN').astype(int)
    
    return df


def train_model(df):
    """Train the enhanced fraud detection model"""
    feature_columns = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'avg_amount', 'transaction_count',
        'amount_std', 'amount_to_balance_ratio', 'amount_zscore', 'is_round_amount',
        'balance_drained', 'dest_transaction_count', 'is_new_dest', 'hourly_txn_count',
        'balance_change_ratio', 'unique_destinations', 'unique_senders_to_dest',
        'deviation_flag', 'high_velocity_flag', 'suspicious_time_flag',
        'txn_frequency_per_day', 'amount_risk_score', 'behavioral_risk_score',
        'type_DEBIT', 'type_TRANSFER', 'type_CASH_OUT', 'type_PAYMENT', 'type_CASH_IN'
    ]
    
    available_features = [col for col in feature_columns if col in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': available_features,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'trained_at': datetime.now().isoformat()
    }
    joblib.dump(model_data, 'fraud_detection_model.pkl')
    return model_data


# ----------------- PREDICT FILE ENDPOINT -----------------
@app.route("/predict_file", methods=["POST"])
def predict_file():
    global model_data
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        return jsonify({"error": "Unsupported file format. Please upload CSV or Excel."}), 400

    # --- Preprocessing for 'type' column ---
    if "type" in df.columns:
        df['type_DEBIT'] = (df['type'] == 'DEBIT').astype(int)
        df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
        df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
        df['type_PAYMENT'] = (df['type'] == 'PAYMENT').astype(int)
        df['type_CASH_IN'] = (df['type'] == 'CASH_IN').astype(int)

    # Ensure all expected features exist
    for col in model_data["feature_columns"]:
        if col not in df.columns:
            df[col] = 0

    # Keep a copy of original data for output
    original_data = df.copy()

    # Select only features for model
    X = df[model_data["feature_columns"]].fillna(0)

    # Scale + predict
    X_scaled = model_data["scaler"].transform(X.values)
    probabilities = model_data["model"].predict_proba(X_scaled)[:, 1]

    results = []
    for i, proba in enumerate(probabilities):
        prediction = "suspicious" if proba > 0.5 else "normal"
        row_result = {
            "row": int(i + 1),
            "prediction": prediction,
            "risk_score": float(proba * 100)
        }

        # Add original values (only key columns)
        if "amount" in original_data.columns:
            row_result["amount"] = float(original_data.iloc[i]["amount"])
        if "type" in original_data.columns:
            row_result["type"] = str(original_data.iloc[i]["type"])

        results.append(row_result)

    return jsonify({"total": len(results), "results": results})




# ----------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Fraud Detection API is running. Use POST /predict_file to upload a file."})
#-----------------MAIN------------------------
if _name_ == "_main_":
    df = load_and_preprocess_data()
    model_data = train_model(df)
    print("Model trained. Starting Flask API...")
    app.run(host="127.0.0.1", port=5000, debug=True)