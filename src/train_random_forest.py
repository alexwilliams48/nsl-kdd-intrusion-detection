"""
Random Forest Classifier for NSL-KDD Intrusion Detection
Trains model WITHOUT data leakage features for honest accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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


def identify_feature_types(df):
   
    # Identify which features are numeric vs categorical
    
    # Returns: tuple: (numeric_features, categorical_features)
    
    # Get all feature columns (exclude labels)
    feature_cols = [col for col in df.columns 
                   if col not in ['attack_type', 'difficulty_level', 
                                  'attack_category', 'is_attack']]
    
    # Separate numeric and categorical
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Categorical: {categorical_features}")
    
    return numeric_features, categorical_features


def remove_data_leakage_features(df):
    
    # Remove features that cause data leakage
    
    # These features use future information not available at prediction time
    
    # Data leakage features (use future information)
    leakage_features = [
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
    ]
    
    # Remove labels and useless features
    columns_to_drop = leakage_features + [
        'attack_type',      
        'difficulty_level', 
        'num_outbound_cmds' 
    ]
    
    # Drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    print("\n" + "="*80)
    print("REMOVING DATA LEAKAGE FEATURES")
    print("="*80)
    print(f"Dropping {len(columns_to_drop)} features:")
    for col in columns_to_drop:
        print(f"  - {col}")
    
    df_clean = df.drop(columns=columns_to_drop)
    
    print(f"\nOriginal features: {len(df.columns)}")
    print(f"Cleaned features: {len(df_clean.columns)}")
    
    return df_clean


def encode_categorical_features(train_df, test_df, categorical_features):
    
    #Convert categorical features to numeric using Label Encoding
    
    #Args:
        # train_df: Training dataframe
        # test_df: Test dataframe
        # categorical_features: List of categorical column names
        
    #Returns tuple: (train_df_encoded, test_df_encoded, encoders)
    
    print("\n" + "="*80)
    print("ENCODING CATEGORICAL FEATURES")
    print("="*80)
    
    encoders = {}
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    for feature in categorical_features:
        print(f"Encoding: {feature}")        
        
        encoder = LabelEncoder()        
        
        encoder.fit(train_df[feature])        
        
        train_encoded[feature] = encoder.transform(train_df[feature])
        
        # Handles unseen categories in test set
        test_encoded[feature] = test_df[feature].map(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
        )
        
        encoders[feature] = encoder
        
        print(f"  Unique values: {len(encoder.classes_)}")
    
    return train_encoded, test_encoded, encoders


def prepare_data_for_training(train_df, test_df):
    
    # Complete data preparation pipeline
    
    # Returns: tuple: (X_train, X_test, y_train, y_test)
   
    print("\n" + "="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)
    
    # Remove data leakage features
    train_clean = remove_data_leakage_features(train_df)
    test_clean = remove_data_leakage_features(test_df)
    
    # Separate features and labels    
    X_train = train_clean.drop(columns=['attack_category', 'is_attack'])
    X_test = test_clean.drop(columns=['attack_category', 'is_attack'])
    
    y_train = train_df['is_attack']
    y_test = test_df['is_attack']
    
    # Identify numerical features  

    numeric_features, categorical_features = identify_feature_types(X_train)
    
    # Encode categorical features
    X_train_encoded, X_test_encoded, encoders = encode_categorical_features(
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
    
    # Train Random Forest classifier
    
    # Returns: 
    # tuple: (model, training_time)
   
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80)
    
    # Create Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,       # 100 trees
        max_depth=20,           # Maximum depth of each tree
        min_samples_split=10,   # Minimum samples to split a node
        min_samples_leaf=5,     # Minimum samples in a leaf
        random_state=42,        # For reproducibility
        n_jobs=-1,              # Use all CPU cores
        verbose=1               # Show progress
    )
    
    print("Model parameters:")
    print(f"  Trees: {rf_model.n_estimators}")
    print(f"  Max depth: {rf_model.max_depth}")
    print(f"  Min samples split: {rf_model.min_samples_split}")    
    
    print("\nTraining started...")
    start_time = time.time()
    
    rf_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {training_time:.2f} seconds")
    
    return rf_model, training_time


def evaluate_model(model, X_train, X_test, y_train, y_test):
    
    # Evaluate model performance on train and test sets
    
    # Returns: dict: Performance metrics
   
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
    
    # Print results
    print("\nTRAINING SET PERFORMANCE:")
    print(f"  Accuracy: {metrics['train_accuracy']:.4f} ({metrics['train_accuracy']*100:.2f}%)")
    
    print("\nTEST SET PERFORMANCE:")
    print(f"  Accuracy:  {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    
    # Classification report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Normal', 'Attack'],
                                digits=4))
    
    return metrics, y_test_pred


def plot_confusion_matrix(y_test, y_pred):
    
    # Plots Confusion matrix 
    
    print("\nGenerating confusion matrix...")    
    
    cm = confusion_matrix(y_test, y_pred)    
    
    fig, ax = plt.subplots(figsize=(8, 6))    
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Random Forest\n(Without Data Leakage Features)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)    
    
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(1, -0.3, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
             ha='center', fontsize=11, transform=ax.transAxes)
    
    plt.tight_layout()    
    
    print("✓ Displaying confusion matrix (close window to continue)")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    
    # Plot feature importance 
   
    print("\nGenerating feature importance plot...")
    
    # Get feature importance
    importance = model.feature_importances_    
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Get top N features
    top_features = importance_df.head(top_n)    
   
    fig, ax = plt.subplots(figsize=(10, 8))
    
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
    
    plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features - Random Forest\n(Without Data Leakage)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()    
    
    for i, v in enumerate(top_features['importance']):
        plt.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    
    print("✓ Displaying feature importance (close window to continue)")
    plt.show()
    
    return importance_df


def save_results_summary(metrics, training_time, importance_df):
    
    # Print results summary 
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    summary = f"""
Model Configuration:
  Algorithm: Random Forest Classifier
  Number of Trees: 100
  Max Depth: 20
  Data Leakage Features: REMOVED 
  
Training Time: {training_time:.2f} seconds

Performance Metrics (Test Set):
  Accuracy:  {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)
  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)

Training Set Accuracy: {metrics['train_accuracy']:.4f} ({metrics['train_accuracy']*100:.2f}%)

Top 10 Most Important Features:
"""
    
    for idx, (i, row) in enumerate(importance_df.head(10).iterrows(), 1):
        summary += f"  {idx}. {row['feature']:25s} {row['importance']:.4f}\n"
    
    summary += f"""
NOTES:
- Data leakage features (dst_host_*) were removed
"""    
    print(summary)


if __name__ == "__main__":
    print("="*80)
    print("RANDOM FOREST INTRUSION DETECTION SYSTEM")
    print("NSL-KDD Dataset - Without Data Leakage Features")
    print("="*80)
    
    
    print("\n Loading data...")
    train_df = load_train_data()
    test_df = load_test_data()
    
    
    train_df = create_labels(train_df)
    test_df = create_labels(test_df)
    
    
    print("\n Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data_for_training(train_df, test_df)
    
    
    print("\n Training Random Forest...")
    model, training_time = train_random_forest(X_train, y_train)
    
    
    print("\n Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    
    print("\n Creating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    importance_df = plot_feature_importance(model, X_train.columns, top_n=20)
    
    
    print("\n Results summary...")
    save_results_summary(metrics, training_time, importance_df)
    
    print("\n" + "="*80)
    print("✓ COMPLETE! Random Forest model trained and evaluated.")
    print("="*80)
    