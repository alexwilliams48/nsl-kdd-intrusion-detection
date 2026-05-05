
# One-Class SVM for Novel Attack Detection
# Trains ONLY on normal traffic (unsupervised)


import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    
    # Scale features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_one_class_svm(X_train, y_train):
    # Train ONLY on normal traffic
    print("\n" + "="*80)
    print("ONE-CLASS SVM - UNSUPERVISED ANOMALY DETECTION")
    print("="*80)
    
    # Get ONLY normal traffic
    X_normal = X_train[y_train == 0]
    
    print(f"\nTraining on NORMAL traffic only:")
    print(f"  Normal samples: {len(X_normal):,}")
    print(f"  (Ignoring {(y_train == 1).sum():,} attack samples)")
    
    # Subsample for speed 
    sample_size = min(10000, len(X_normal))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_normal), sample_size, replace=False)
    X_normal_sample = X_normal[sample_idx]
    
    print(f"  Using {sample_size:,} samples for training (SVM is slow on large data)")
    
    # Train One-Class SVM
    model = OneClassSVM(
        kernel='rbf',
        nu=0.1,  # Expected outlier fraction
        gamma='scale'
    )
    
    print("\nTraining One-Class SVM...")   
    start_time = time.time()
    model.fit(X_normal_sample)
    training_time = time.time() - start_time
    
    print(f"✓ Training complete in {training_time:.2f} seconds")
    
    return model, training_time

def evaluate(model, X_test, y_test):
    # Evaluate anomaly detection 
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    # Predict: +1 = normal, -1 = anomaly
    predictions = model.predict(X_test)
    
    # Convert to binary
    y_pred = (predictions == -1).astype(int)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print("\nONE-CLASS SVM RESULTS:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")  
   
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], digits=4))
    
    return metrics

if __name__ == "__main__":
    print("="*80)
    print("ONE-CLASS SVM - NOVEL ATTACK DETECTION")
    print("="*80)    
    
    train_df = load_train_data()
    test_df = load_test_data()
    train_df = create_labels(train_df)
    test_df = create_labels(test_df)    
    
    X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)    
   
    model, training_time = train_one_class_svm(X_train, y_train)    
    
    metrics = evaluate(model, X_test, y_test)
    
   