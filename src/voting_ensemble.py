# Voting Ensemble: XGBoost + Random Forest + Isolation Forest + One-Class SVM
# Majority vote determines final prediction

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

def train_all_models(X_train, y_train):
    
    print("\n" + "="*80)
    print("TRAINING ALL 4 MODELS FOR ENSEMBLE")
    print("="*80)
    
    models = {}
    
    # XGBoost
    print("\n1. Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models['xgboost'] = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    models['xgboost'].fit(X_train, y_train)
    print("   ✓ XGBoost trained")
    
    # Random Forest
    print("\n2. Training Random Forest...")
    models['random_forest'] = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    models['random_forest'].fit(X_train, y_train)
    print("   ✓ Random Forest trained")
    
    # Isolation Forest (trained only on normal traffic)
    print("\n3. Training Isolation Forest (on normal traffic)...")
    X_normal = X_train[y_train == 0]
    models['isolation_forest'] = IsolationForest(
        n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1
    )
    models['isolation_forest'].fit(X_normal)
    print("   ✓ Isolation Forest trained")
    
    # One-Class SVM (subsampled for speed)
    print("\n4. Training One-Class SVM (subsample for speed)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_normal_scaled = X_train_scaled[y_train == 0]
    sample_size = min(10000, len(X_normal_scaled))
    sample_idx = np.random.choice(len(X_normal_scaled), sample_size, replace=False)
    models['one_class_svm'] = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
    models['one_class_svm'].fit(X_normal_scaled[sample_idx])
    print("   ✓ One-Class SVM trained")
    
    return models, scaler

def voting_ensemble_predict(models, scaler, X, threshold=2):
    
    # threshold=2: At least 2/4 models must agree for attack classification
    # threshold=3: At least 3/4 models must agree for attack classification
    
    predictions = np.zeros(len(X))
    
    # Get predictions from all models
    xgb_pred = models['xgboost'].predict(X)
    rf_pred = models['random_forest'].predict(X)
    
    # Isolation Forest: -1 (outlier/attack) → 1, +1 (normal) → 0
    iso_pred = models['isolation_forest'].predict(X)
    iso_pred = (iso_pred == -1).astype(int)
    
    # One-Class SVM: -1 (outlier/attack) → 1, +1 (normal) → 0
    X_scaled = scaler.transform(X)
    svm_pred = models['one_class_svm'].predict(X_scaled)
    svm_pred = (svm_pred == -1).astype(int)
    
    # Count votes for each sample
    votes = xgb_pred + rf_pred + iso_pred + svm_pred
    
    # Apply threshold
    predictions = (votes >= threshold).astype(int)
    
    return predictions, votes

def evaluate_ensemble(models, scaler, X_test, y_test, threshold=2):
    
    print("\n" + "="*80)
    print(f"VOTING ENSEMBLE EVALUATION (threshold={threshold})")
    print("="*80)
    
    print(f"\nVoting Rule: Attack if >= {threshold} models agree")
    
    start_time = time.time()
    predictions, votes = voting_ensemble_predict(models, scaler, X_test, threshold)
    inference_time = time.time() - start_time
    
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions)
    }
    
    # Vote distribution
    print(f"\nVote Distribution:")
    for v in range(5):
        count = (votes == v).sum()
        pct = count / len(votes) * 100
        print(f"  {v} votes (attack): {count:,} samples ({pct:.1f}%)")
    
    print(f"\nInference time: {inference_time:.3f}s ({len(X_test)/inference_time:.0f} samples/sec)")
    
    print(f"\nENSEMBLE RESULTS:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    
    return metrics


if __name__ == "__main__":
    print("="*80)
    print("VOTING ENSEMBLE - ALL 4 MODELS")
    print("="*80)    
    
    train_df = load_train_data()
    test_df = load_test_data()
    train_df = create_labels(train_df)
    test_df = create_labels(test_df)    
    
    train_clean = remove_data_leakage_features(train_df)
    test_clean = remove_data_leakage_features(test_df)
    
    X_train = encode_categorical(train_clean.drop(columns=['attack_category', 'is_attack']))
    X_test = encode_categorical(test_clean.drop(columns=['attack_category', 'is_attack']))
    y_train = train_df['is_attack']
    y_test = test_df['is_attack']
    
    # Train all models
    models, scaler = train_all_models(X_train, y_train)
    
    # Evaluate with different voting thresholds
    metrics_t2 = evaluate_ensemble(models, scaler, X_test, y_test, threshold=2)
    metrics_t3 = evaluate_ensemble(models, scaler, X_test, y_test, threshold=3)    
    
    
    print("\n" + "="*80)
    print("✓ VOTING ENSEMBLE COMPLETE!")
    print("="*80)