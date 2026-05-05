import pandas as pd

def get_column_names():
    
    ##Returns NSL-KDD column names    
    ##Total: 43 columns (41 features + 2 labels)
    ##Source: http://www.unb.ca/cic/datasets/nsl.html
    
    columns = [
        # Basic features
        'duration', 'protocol_type', 'service', 'flag',
        
        # Content features
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login',
        
        # Traffic features  
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate',
        
        # Host-based features
        'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        
        # Labels
        'attack_type', 'difficulty_level'
    ]
    
    return columns

def load_train_data():
    
    ##Load NSL-KDD training data
    columns = get_column_names()
    
    print("Loading training data...")
    train_df = pd.read_csv(
        'data/KDDTrain+.txt',
        names=columns,      
        header=None         
    )
    print(f"✓ Loaded {len(train_df):,} training samples")
    
    return train_df

def load_test_data():
    
    ##Load NSL-KDD test data  
    columns = get_column_names()
    
    print("Loading test data...")
    test_df = pd.read_csv(
        'data/KDDTest+.txt',
        names=columns,
        header=None
    )
    print(f"✓ Loaded {len(test_df):,} test samples")
    
    return test_df


def get_attack_mapping():
    
    # Map specific attack types to broader categories
    #     
    # Returns: dict: Mapping of attack_type -> attack_category
    
    attack_map = {
        # DoS (Denial of Service) attacks
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
        'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos',
        'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
        
        # Probe (Reconnaissance) attacks
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
        'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
        
        # R2L (Remote to Local) attacks
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
        'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l',
        'warezclient': 'r2l', 'warezmaster': 'r2l', 'sendmail': 'r2l',
        'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
        'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
        
        # U2R (User to Root) attacks
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
        'rootkit': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r',
        'sqlattack': 'u2r', 'xterm': 'u2r',
        
        # Normal traffic
        'normal': 'normal'
    }
    
    return attack_map


def create_labels(df):
    
    ##Add attack_category and binary is_attack labels    
    ##Args: df (pd.DataFrame): Dataframe with 'attack_type' column        
   
    attack_map = get_attack_mapping()
    
    # Add attack category (dos, probe, r2l, u2r, normal)
    df['attack_category'] = df['attack_type'].map(attack_map)
    
    # Add binary label (0 = normal, 1 = attack)
    df['is_attack'] = (df['attack_category'] != 'normal').astype(int)
    
    return df


def show_data_info(train_df, test_df):
    #Print dataset statistics and sample data
    
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    
    print(f"\nTraining samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Total features: {len(train_df.columns)}")
    
    print("\n" + "="*80)
    print("ATTACK DISTRIBUTION (Training)")
    print("="*80)
    print(train_df['attack_category'].value_counts())
    
    print("\n" + "="*80)
    print("BINARY CLASSIFICATION")
    print("="*80)
    normal_count = (train_df['is_attack'] == 0).sum()
    attack_count = (train_df['is_attack'] == 1).sum()
    
    print(f"Normal: {normal_count:,} ({normal_count/len(train_df)*100:.1f}%)")
    print(f"Attack: {attack_count:,} ({attack_count/len(train_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("SAMPLE DATA (First 5 Rows)")
    print("="*80)
    display_cols = ['duration', 'protocol_type', 'service', 'src_bytes', 
                    'dst_bytes', 'attack_type', 'attack_category', 'is_attack']
    print(train_df[display_cols].head())




if __name__ == "__main__":
    
    train_df = load_train_data()
    test_df = load_test_data()    
    
    train_df = create_labels(train_df)
    test_df = create_labels(test_df)    
    
    show_data_info(train_df, test_df)
    
    print("Data loading complete")