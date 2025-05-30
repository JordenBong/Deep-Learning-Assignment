## Data Preprocesss
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, StandardScaler



def preprocess_dataset(filename): 

    if "train" in filename:
        train = True
        test = False
    else:
        test = True
        train = False
    
    # Load dataset
    df = pd.read_csv(filename) # filename = "test_merged_df.csv"
    
    ## Preprocess for sequence data
    # Drop sensitive features that are likely unique per bidder
    df = df.drop(['payment_account', 'address'], axis=1)
    
    # Encode categorical features
    categorical_cols = ['auction', 'merchandise', 'device', 'country']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        if train:
            le.train
        df[col] = le.transform(df[col])
        label_encoders[col] = le
    
    # Frequency encoding for high-cardinality features (ip, url)
    for col in ['ip', 'url']:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)
    df = df.drop(['ip', 'url'], axis=1)
    
    # Normalize numerical feature (time)
    scaler = StandardScaler()
    if train:
        scaler.fit(df[['time']])
    df['time'] = scaler.transform(df[['time']])

    # Step 3: Create Sequences
    # Group bids by bidder_id and sort by time
    df = df.sort_values(['bidder_id', 'time'])
    sequences = []
    labels = []
    bidder_ids = []
    feature_cols = ['auction', 'merchandise', 'device', 'time', 'country', 'ip_freq', 'url_freq']
    
    for bidder_id, group in df.groupby('bidder_id'):
        # Extract features for the sequence
        seq = group[feature_cols].values
        sequences.append(seq)
        bidder_ids.append(bidder_id)
        if train:
            # Outcome is the same for all bids of a bidder
            labels.append(group['outcome'].iloc[0])

    # bidder_id (merged with prediction, for submmision purpose)
    


    # Step 4: Pad Sequences
    # Pad sequences to the same length (use max length or a reasonable fixed length)
    max_len = min(max(len(seq) for seq in sequences), 200)  # Cap at 100 for efficiency
    X = pad_sequences(sequences, maxlen=max_len, padding='post', dtype='float32')
    bidder_ids = np.array(bidder_ids) 

    if train:
        y = np.array(labels)

    # Return X (and y) based on test or train
    if test:
        return X, bidder_ids
    else: 
        return X, y, bidder_ids
    