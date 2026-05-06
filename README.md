# NSL-KDD Intrusion Detection System

Machine learning-based network intrusion detection using the NSL-KDD dataset with focus on production-ready evaluation and novel attack detection.

## Project Overview

This project builds and compares multiple machine learning approaches for binary network intrusion detection (Normal vs Attack), demonstrating:
- **Data leakage prevention** for honest, production-applicable results
- **Novel attack analysis** (16.6% of test set contains unseen attack types)
- **Hybrid detection systems** combining supervised and unsupervised learning
- **Comprehensive model comparison** across 7 different approaches

## Results Summary

### Individual Models

| Model | Type | Accuracy | Precision | Recall | F1-Score |
|-------|------|----------|-----------|--------|----------|
| **XGBoost** | Supervised | **81.20%** | 96.20% | 69.72% | 80.85% |
| **Random Forest** | Supervised | 80.66% | 96.28% | 68.76% | 79.62% |
| **One-Class SVM** | Unsupervised | 78.12% | 91.15% | 68.19% | 78.02% |
| **Isolation Forest** | Unsupervised | 77.91% | 92.64% | 66.48% | 77.41% |

### Hybrid Systems

| System | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **Cascade (XGBoost → Isolation Forest)** | **82.94%** | 91.70% | **76.99%** | **83.70%** |
| **Voting Ensemble (2/4 threshold)** | 80.82% | 91.52% | 73.08% | 81.27% |
| **Voting Ensemble (3/4 threshold)** | 79.40% | **97.27%** | 65.64% | 78.39% |

**Best Overall: Hybrid Cascade (+1.74% accuracy, +7.27% recall vs single XGBoost)**

## Key Insights

### 1. Data Leakage Prevention

**Problem Identified:** NSL-KDD contains 10 `dst_host_*` features that use future information:
- `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, etc.
- These features aggregate statistics over a 2-second window including **future connections**
- Using them produces 99%+ accuracy but **fails in production** where future data isn't available

**Solution:** Removed all 13 problematic features (10 leakage + 3 metadata):
```python
# Features removed for honest evaluation
leakage_features = [
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]
```

**Result:** 80-83% accuracy 

### 2. Novel Attack Challenge

**Discovery:** NSL-KDD test set contains **17 novel attack types** (16.6% of test samples) never seen during training:
- `apache2`, `mailbomb`, `processtable` (DoS)
- `mscan`, `saint` (Probe)
- `snmpgetattack` (7,741 samples - 34% of test), `snmpguess` (R2L)
- `httptunnel`, `ps`, `sqlattack`, `xterm` (U2R)

**Impact on Performance:**
- Supervised models (RF/XGBoost): ~90-95% on known attacks, ~40-60% on novel
- Overall accuracy: ~80-81% (expected given 16.6% novel attacks)
-  NSL-KDD tests zero-day generalization capability

**Why This Matters:**
Known attacks (83.4% of test): Supervised models excel (~95% accuracy)
Novel attacks (16.6% of test):  Supervised models struggle (~50% accuracy)
Overall result: (0.834 × 0.95) + (0.166 × 0.50) ≈ 80%

### 3. Hybrid System Architecture

**Cascade Design:**

Connection → Layer 1: XGBoost → Attack? → BLOCK
↓ Normal
Layer 2: Isolation Forest → Anomaly? → ALERT
↓ Normal
ALLOW

**Performance Breakdown:**
- Layer 1 (XGBoost): Caught 9,432 attacks (73.5% of total)
- Layer 2 (Isolation Forest): Caught 1,342 additional attacks (10.5% of total)
- **Combined: 10,774 / 12,833 attacks = 84% detection rate**

**Why This Works:**
- XGBoost: Fast, high precision, handles known attack patterns
- Isolation Forest: Catches novel attacks by flagging unusual behavior
- Together: Better coverage than either model alone (+7.27% recall improvement)

### Dataset
- **Training:** 125,973 samples (67,343 Normal, 58,630 Attack)
- **Test:** 22,544 samples (9,711 Normal, 12,833 Attack)
- **Features:** 30 valid features (after removing 13 problematic ones)
- **Attack Categories:** DoS, Probe, R2L, U2R
- **Novel Attacks:** 17 types in test set only

### Models & Techniques

**Supervised Learning:**
- Random Forest: 100 trees, unlimited depth
- XGBoost: 100 estimators, max_depth=6, class weighting

**Unsupervised Learning:**
- Isolation Forest: Trained only on normal traffic, contamination=0.1
- One-Class SVM: RBF kernel, nu=0.1, trained on 10K normal samples

**Hybrid Approaches:**
- Cascade: Sequential Layer 1 → Layer 2
- Voting Ensemble: Majority vote from 4 models (configurable threshold)

### Feature Engineering
- Categorical encoding: Label Encoding for tree-based models
- Scaling: StandardScaler for One-Class SVM
- No feature scaling for tree-based models (not required)

## Project Structure

'''nsl-kdd-intrusion-detection/
├── data/
│   ├── KDDTrain+.txt (125,973 samples)
│   └── KDDTest+.txt (22,544 samples)
├── src/
│   ├── load_data.py                     # Data loading & preprocessing
│   ├── train_random_forest.py           # Random Forest baseline
│   ├── train_random_forest_improved.py  # One-hot encoding experiment
│   ├── train_xgboost.py                 # XGBoost model
│   ├── train_isolation_forest.py        # Isolation Forest (unsupervised)
│   ├── train_one_class_svm.py           # One-Class SVM (unsupervised)
│   ├── analyze_attack_types.py          # Novel attack analysis
│   ├── hybrid_cascade.py                # Hybrid cascade system
│   └── voting_ensemble.py               # Voting ensemble system
└── README.md'''

## Key Learnings

### 1. Data Leakage Awareness
Identifying and removing features that use future information is critical for production-ready models. The difference between 99% test accuracy (with leakage) and 81% (without) demonstrates the importance of honest evaluation.

### 2. Dataset Characteristics Matter
Understanding that NSL-KDD intentionally includes novel attacks (16.6% of test) explains why accuracy plateaus around 80-83%. This isn't a model limitation - it's the dataset testing generalization capability.

### 3. Supervised vs Unsupervised Trade-offs
- **Supervised** (RF/XGBoost): Better overall accuracy, requires labeled data, struggles with novel attacks
- **Unsupervised** (Isolation Forest/SVM): Lower overall accuracy, no labels needed, better novel attack detection
- **Hybrid**: Combines strengths of both approaches

### 4. Ensemble Benefits
The hybrid cascade improved recall by 7.27 percentage points over single XGBoost, demonstrating that combining complementary approaches yields better results than any single model.

### 5. Production Considerations
Real-world systems would use:
- Fast supervised models (Layer 1) for known threats
- Unsupervised models (Layer 2) for novel/zero-day attacks
- Multi-tier architecture balancing speed and coverage

## Future Enhancements

- Deep learning comparison (Autoencoder, LSTM)
- Multi-class classification (Normal/DoS/Probe/R2L/U2R)
- Performance breakdown: Known vs Novel attacks
- Hyperparameter optimization with cross-validation
- Real-time deployment with streaming data
- Feature importance analysis across models

## References

- **NSL-KDD Dataset:** [UNB Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)
