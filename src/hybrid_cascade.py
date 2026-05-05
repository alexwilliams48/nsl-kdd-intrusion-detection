
# Hybrid Cascade Detection: XGBoost + Isolation Forest
# Layer 1: XGBoost catches known attacks 
# Layer 2: Isolation Forest catches novel attacks 


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time

from load_data import load_train_data, load_test_data, create_labels

def remove_data_leakage_features(df):
    leakage_features = [
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    ]
    columns_to_drop = leakage_features + ['attack_type', 'difficulty_level', 'num_outbound_cmds']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=columns_to_drop)

def encode_categorical(df):
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = df.copy()
    for feature in categorical_features:
        encoder = LabelEncoder()
        df_encoded[feature] = encoder.fit_transform(df[feature])
    return df_encoded

def prepare_data(train_df, test_df):
    train_clean = remove_data_leakage_features(train_df)
    test_clean = remove_data_leakage_features(test_df)
    
    X_train = train_clean.drop(columns=['attack_category', 'is_attack'])
    X_test = test_clean.drop(columns=['attack_category', 'is_attack'])
    y_train = train_df['is_attack']
    y_test = test_df['is_attack']
    
    X_train_enc = encode_categorical(X_train)
    X_test_enc = encode_categorical(X_test)
    
    return X_train_enc, X_test_enc, y_train, y_test

def train_models(X_train, y_train):
    # Train both XGBoost and Isolation Forest
    print("\n" + "="*80)
    print("TRAINING HYBRID CASCADE MODELS")
    print("="*80)
    
    # Layer 1: XGBoost 
    print("\nLayer 1: Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    print("✓ XGBoost trained")
    
    # Layer 2: Isolation Forest (only normal traffic)
    print("\nLayer 2: Training Isolation Forest (on normal traffic only)...")
    X_normal = X_train[y_train == 0]
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_normal)
    print("✓ Isolation Forest trained")
    
    return xgb_model, iso_model

def hybrid_cascade_predict(xgb_model, iso_model, X):
    
    #Cascade prediction:
    # 1. XGBoost predicts first
    # 2. If normal, check Isolation Forest
    
    predictions = np.zeros(len(X))
    layer1_catches = 0
    layer2_catches = 0
    
    # Layer 1: XGBoost
    xgb_pred = xgb_model.predict(X)
    
    for i in range(len(X)):
        if xgb_pred[i] == 1:  # XGBoost says attack
            predictions[i] = 1
            layer1_catches += 1
        else:  # XGBoost says normal, check Layer 2
            iso_pred = iso_model.predict(X[i:i+1])
            if iso_pred[0] == -1:  # Isolation Forest says anomaly
                predictions[i] = 1
                layer2_catches += 1
            else:
                predictions[i] = 0
    
    return predictions, layer1_catches, layer2_catches

def evaluate_hybrid(xgb_model, iso_model, X_test, y_test):
    # Evaluate cascade system
    print("\n" + "="*80)
    print("HYBRID CASCADE EVALUATION")
    print("="*80)
    
    print("\nRunning cascade detection...")
    start_time = time.time()
    predictions, layer1_catches, layer2_catches = hybrid_cascade_predict(
        xgb_model, iso_model, X_test
    )
    inference_time = time.time() - start_time
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions)
    }
    
    total_attacks = (y_test == 1).sum()
    
    print(f"\nDetection Breakdown:")
    print(f"  Layer 1 (XGBoost) caught:       {layer1_catches:,} attacks")
    print(f"  Layer 2 (Isolation Forest) caught: {layer2_catches:,} attacks")
    print(f"  Total attacks in test:          {total_attacks:,}")
    print(f"  Inference time:                 {inference_time:.3f}s ({len(X_test)/inference_time:.0f} samples/sec)")
    
    print(f"\nHYBRID CASCADE RESULTS:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    
    return metrics

if __name__ == "__main__":
    print("="*80)
    print("HYBRID CASCADE DETECTION SYSTEM")
    print("XGBoost → Isolation Forest")
    print("="*80)    
    
    train_df = load_train_data()
    test_df = load_test_data()
    train_df = create_labels(train_df)
    test_df = create_labels(test_df)
    
    # Prepare
    X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)
    
    # Train both models
    xgb_model, iso_model = train_models(X_train, y_train)
    
    # Evaluate hybrid system
    hybrid_metrics = evaluate_hybrid(xgb_model, iso_model, X_test, y_test)   
    
    
