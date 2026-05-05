
# XGBoost Classifier for NSL-KDD


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time

from load_data import load_train_data, load_test_data, create_labels

def remove_data_leakage_features(df):
    # Remove dst_host_* features
    leakage_features = [
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    ]
    columns_to_drop = leakage_features + ['attack_type', 'difficulty_level', 'num_outbound_cmds']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=columns_to_drop)

def encode_categorical(train_df, test_df):
    # Label encode categorical features 
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
    
    encoders = {}
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    for feature in categorical_features:
        encoder = LabelEncoder()
        encoder.fit(train_df[feature])
        
        train_encoded[feature] = encoder.transform(train_df[feature])
        test_encoded[feature] = test_df[feature].map(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
        )
        encoders[feature] = encoder
    
    return train_encoded, test_encoded

def prepare_data(train_df, test_df):
    # Prepare data for XGBoost
    train_clean = remove_data_leakage_features(train_df)
    test_clean = remove_data_leakage_features(test_df)
    
    X_train = train_clean.drop(columns=['attack_category', 'is_attack'])
    X_test = test_clean.drop(columns=['attack_category', 'is_attack'])
    y_train = train_df['is_attack']
    y_test = test_df['is_attack']
    
    X_train_enc, X_test_enc = encode_categorical(X_train, X_test)
    
    print(f"X_train: {X_train_enc.shape}")
    print(f"X_test: {X_test_enc.shape}")
    
    return X_train_enc, X_test_enc, y_train, y_test

def train_xgboost(X_train, y_train):
    # Train XGBoost model
    print("\nTraining XGBoost...")
    
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Training complete in {training_time:.2f} seconds")
    
    return model, training_time

def evaluate(model, X_train, X_test, y_train, y_test):
    # Evaluate and compare with RF baseline
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred)
    }
    
    print("\n" + "="*80)
    print("XGBOOST RESULTS")
    print("="*80)
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f} ({metrics['train_accuracy']*100:.2f}%)")
    print(f"Test Accuracy:     {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
    print(f"Precision:         {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:            {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:          {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")  
 
    
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_test, y_test_pred, target_names=['Normal', 'Attack'], digits=4))
    
    return metrics

if __name__ == "__main__":
    print("="*80)
    print("XGBOOST INTRUSION DETECTION")
    print("="*80)
    
    
    train_df = load_train_data()
    test_df = load_test_data()
    train_df = create_labels(train_df)
    test_df = create_labels(test_df)    
    
    X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)    
   
    model, training_time = train_xgboost(X_train, y_train)    
    
    metrics = evaluate(model, X_train, X_test, y_train, y_test)
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)