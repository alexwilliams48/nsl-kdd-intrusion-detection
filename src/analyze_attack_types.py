
# Analyis of attack type distribution in train vs test sets


import pandas as pd
from load_data import load_train_data, load_test_data, create_labels

print("Loading data...")
train_df = load_train_data()
test_df = load_test_data()
train_df = create_labels(train_df)
test_df = create_labels(test_df)


print("ATTACK TYPE ANALYSIS: TRAIN vs TEST")

train_attacks = set(train_df['attack_type'].unique())
test_attacks = set(test_df['attack_type'].unique())

print(f"\nTraining set attack types: {len(train_attacks)}")
print(f"Test set attack types: {len(test_attacks)}")

# Find novel attacks in test set
novel_attacks = test_attacks - train_attacks
common_attacks = test_attacks & train_attacks


print("Attacks in test set that were never seen in training:")

print(f"Count: {len(novel_attacks)}\n")

if novel_attacks:
    for attack in sorted(novel_attacks):
        count = (test_df['attack_type'] == attack).sum()
        pct = count / len(test_df) * 100
        category = test_df[test_df['attack_type'] == attack]['attack_category'].iloc[0]
        print(f"  {attack:20s} ({category:8s}): {count:5d} samples ({pct:5.2f}%)")
    
    total_novel = test_df[test_df['attack_type'].isin(novel_attacks)].shape[0]
    novel_pct = total_novel / len(test_df) * 100
    
    print(f"\nTotal novel attack samples: {total_novel:,} ({novel_pct:.2f}% of test set)")


print("COMMON ATTACKS (In Both Train and Test)")
print(f"Count: {len(common_attacks)}\n")

for attack in sorted(common_attacks):
    train_count = (train_df['attack_type'] == attack).sum()
    test_count = (test_df['attack_type'] == attack).sum()
    print(f"  {attack:20s}: Train={train_count:6d}, Test={test_count:5d}")

