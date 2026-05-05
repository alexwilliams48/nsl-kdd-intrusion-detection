
# Improved Random Forest Classifier for NSL-KDD
# Key improvements:
# 1. One-Hot Encoding instead of Label Encoding (fixes categorical feature issue)
# 2. Tuned hyperparameters (unlimited depth, more flexible splits)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import time

from load_data import load_train_data, load_test_data, create_labels

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def remove_data_leakage_features(df):
    
    # Remove features that cause data leakage
    
    leakage_features = [
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
    ]
    
    columns_to_drop = leakage_features + [
        'attack_type', 'difficulty_level', 'num_outbound_cmds'
    ]
    
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    print("\n" + "="*80)
    print("REMOVING DATA LEAKAGE FEATURES")
    print("="*80)
    print(f"Dropping {len(columns_to_drop)} features")
    
    df_clean = df.drop(columns=columns_to_drop)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Cleaned features: {len(df_clean.columns)}")
    
    return df_clean


def one_hot_encode_categorical(train_df, test_df, categorical_features):
    
    # ONE-HOT ENCODING
    
    # Instead of: service='http' → 0, service='ftp' → 1
    # Creates: service_http=1, service_ftp=0, service_smtp=0, ...
    
    # Prevents Random Forest from thinking categories have order
    
    print("\n" + "="*80)
    print("ONE-HOT ENCODING CATEGORICAL FEATURES")
    print("="*80)
    
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    for feature in categorical_features:
        print(f"\nEncoding: {feature}")
        print(f"  Unique values: {train_df[feature].nunique()}")
        
        # Get all categories from training data
        all_categories = train_df[feature].unique()
        
        # Create binary columns for each category
        for category in all_categories:
            col_name = f"{feature}_{category}"
            
            # Create binary column (1 if matches, 0 otherwise)
            train_encoded[col_name] = (train_df[feature] == category).astype(int)
            test_encoded[col_name] = (test_df[feature] == category).astype(int)
        
        # Drop original categorical column
        train_encoded = train_encoded.drop(columns=[feature])
        test_encoded = test_encoded.drop(columns=[feature])
        
        print(f"  Created {len(all_categories)} binary columns")
    
    print(f"\nTotal features after one-hot encoding: {len(train_encoded.columns)}")
    
    return train_encoded, test_encoded


def prepare_data(train_df, test_df):
    
    # Data preparation pipeline with One-Hot Encoding
    
    print("\n" + "="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)    
    
    # Remove leakage features
    train_clean = remove_data_leakage_features(train_df)
    test_clean = remove_data_leakage_features(test_df)    
    
    # Separate features and labels
    X_train = train_clean.drop(columns=['attack_category', 'is_attack'])
    X_test = test_clean.drop(columns=['attack_category', 'is_attack'])
    
    y_train = train_df['is_attack']
    y_test = test_df['is_attack']
    
    # Identify categorical features
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical features to encode: {categorical_features}")
    
    # One-Hot Encoding
    X_train_encoded, X_test_encoded = one_hot_encode_categorical(
        X_train, X_test, categorical_features
    )
    
    print("\n" + "="*80)
    print("FINAL DATA SHAPES")
    print("="*80)
    print(f"X_train: {X_train_encoded.shape}")
    print(f"X_test: {X_test_encoded.shape}")
    print(f"y_train: {y_train.shape} - Normal: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")
    print(f"y_test: {y_test.shape} - Normal: {(y_test==0).sum()}, Attack: {(y_test==1).sum()}")
    
    return X_train_encoded, X_test_encoded, y_train, y_test


def train_random_forest(X_train, y_train):
    
    # Train Random Forest with optimized hyperparameters
    
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80)
    
    print("\nMODEL CONFIGURATION:")
    print("  ✓ One-Hot Encoding for categorical features")
    print("  ✓ max_depth: None (unlimited depth)")
    print("  ✓ min_samples_split: 5 (flexible splits)")
    print("  ✓ min_samples_leaf: 2 (flexible leaves)")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,         # Unlimited depth
        min_samples_split=5,    # More flexible splits
        min_samples_leaf=2,     # More flexible leaves
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("\nHyperparameters:")
    print(f"  Trees: {rf_model.n_estimators}")
    print(f"  Max depth: {rf_model.max_depth} (unlimited)")
    print(f"  Min samples split: {rf_model.min_samples_split}")
    print(f"  Min samples leaf: {rf_model.min_samples_leaf}")
    
    # Train
    print("\nTraining started...")
    start_time = time.time()
    
    rf_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {training_time:.2f} seconds")
    
    return rf_model, training_time


def evaluate_model(model, X_train, X_test, y_train, y_test):
    
    # Evaluate model performance
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred)
    }
    
    print("\nMODEL PERFORMANCE:")
    print(f"  Training Accuracy: {metrics['train_accuracy']:.4f} ({metrics['train_accuracy']*100:.2f}%)")
    print(f"  Test Accuracy:     {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
    print(f"  Precision:         {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:            {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:          {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    
    
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Normal', 'Attack'],
                                digits=4))
    
    return metrics, y_test_pred


def plot_confusion_matrix(y_test, y_pred):
    
    # Plot confusion matrix
    
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                cbar_kws={'label': 'Count'})
    
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    ax.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    print("✓ Displaying confusion matrix (close window to continue)")
    plt.show()


def plot_metrics_chart(metrics):
    
    # Plot performance metrics
    
    print("\nGenerating metrics chart...")
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        metrics['test_accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(metrics_names, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Random Forest Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}\n({height*100:.2f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    print("✓ Displaying metrics chart (close window to continue)")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    
    # Plot top feature importances
    
    print("\nGenerating feature importance plot...")    
   
    importance = model.feature_importances_    
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(top_n)
       
    fig, ax = plt.subplots(figsize=(10, 8))    
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', 
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    
    for i, v in enumerate(top_features['importance']):
        plt.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    print("✓ Displaying feature importance (close window to continue)")
    plt.show()
    
    return importance_df


if __name__ == "__main__":
    print("="*80)
    print("RANDOM FOREST CLASSIFIER FOR NSL-KDD INTRUSION DETECTION")
    print("="*80)
    
    print("\nKEY FEATURES:")
    print("  ✓ One-Hot Encoding for categorical features")
    print("  ✓ Optimized hyperparameters for better performance")
    print("  ✓ Data leakage features removed")
    
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    train_df = load_train_data()
    test_df = load_test_data()
    train_df = create_labels(train_df)
    test_df = create_labels(test_df)    
    
    print("\n" + "="*80)
    print(" Data Preparation")
    print("="*80)
    X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)
    
    print("\n" + "="*80)
    print("Training Model")
    print("="*80)
    model, training_time = train_random_forest(X_train, y_train)    
    
    print("\n" + "="*80)
    print("Evaluation")
    print("="*80)
    metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Visualizations
    print("\n" + "="*80)
    print("Visualizations")
    print("="*80)
    
    plot_metrics_chart(metrics)
    plot_confusion_matrix(y_test, y_pred)
    importance_df = plot_feature_importance(model, X_train.columns, top_n=20)
    
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
    print(f"  Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:      {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"  Training Time: {training_time:.2f} seconds")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)