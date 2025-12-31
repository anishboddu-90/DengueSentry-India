import pandas as pd
import numpy as np
from src.outbreak_detection import farrington_flexible_label

def load_dengue_data(file_path, label_path=None, city_filter=None):
    df = pd.read_csv(file_path)
    if label_path:
        labels = pd.read_csv(label_path)
        df = pd.merge(df, labels, on=['city', 'year', 'weekofyear'])
    
    # Standardize Date
    date_col = 'week_start_date' if 'week_start_date' in df.columns else 'date'
    df['date'] = pd.to_datetime(df[date_col])
    
    if city_filter and 'city' in df.columns:
        df = df[df['city'] == city_filter]
        
    return df.sort_values('date').ffill().bfill()

def engineer_features(df, cases_col='total_cases', temp_col=None, precip_col=None, method='2sigma', sigma=2.0):
    df = df.copy()
    
    # --- Spike Labeling ---
    if method == '2sigma':
        df['mean_52'] = df[cases_col].rolling(window=52, min_periods=1).mean()
        df['std_52'] = df[cases_col].rolling(window=52, min_periods=1).std()
        df['spike'] = ((df[cases_col] - df['mean_52']) > sigma * df['std_52']).astype(int)
        df['spike_method'] = '2sigma'
    
    elif method == 'farrington':
        df = farrington_flexible_label(df, cases_col=cases_col)
        df['spike'] = df['spike_farrington']  # Rename to standard 'spike'
        df['spike_method'] = 'farrington'
    
    else:
        raise ValueError("method must be '2sigma' or 'farrington'")
    
    # --- Feature Engineering ---
    df['momentum'] = df[cases_col].ewm(span=4, adjust=False).mean() - df[cases_col].ewm(span=12, adjust=False).mean()
    
    features = ['momentum']
    if temp_col and temp_col in df.columns:
        df['temp_lag1'] = df[temp_col].shift(1)
        features.append('temp_lag1')
    if precip_col and precip_col in df.columns:
        df['precip_lag1'] = df[precip_col].shift(1)
        df['precip_lag4'] = df[precip_col].shift(4)
        features.extend(['precip_lag1', 'precip_lag4'])
    
    # Predict NEXT week's spike
    df['target'] = df['spike'].shift(-1)
    
    # Standardize date column for simplicity
    date_col = 'date' if 'date' in df.columns else 'week_start_date'
    df = df.rename(columns={date_col: 'date'})
    
    cols_to_keep = features + ['target', 'date', cases_col, 'spike', 'spike_method']
    df_clean = df.dropna(subset=features + ['target'])[cols_to_keep]
    
    return df_clean.reset_index(drop=True), features

def get_processed_data(file_path, label_path=None, cases_col='total_cases', temp_col=None, precip_col=None, city_filter=None):
    raw_df = load_dengue_data(file_path, label_path, city_filter)
    clean_df, features = engineer_features(raw_df, cases_col, temp_col, precip_col)
    
    print(f"--- Dataset Processed ---")
    print(f"Rows: {len(clean_df)} | Spike Rate: {clean_df['spike'].mean():.2%}")
    return clean_df, features