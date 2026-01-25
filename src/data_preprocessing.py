import pandas as pd
import numpy as np
import logging
import warnings
from typing import Optional, List, Tuple, Dict
from datetime import datetime
import json
import os
from src.weather_engine import fetch_nasa_weather_data
from src.outbreak_detection import detect_outbreak_signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Region-specific configurations
REGION_CONFIGS = {
    "san_juan": { "latitude": 18.4663, "longitude": -66.1057, "climate_zone": "tropical_coastal" },
    "karnataka": { "latitude": 15.3173, "longitude": 75.7139, "climate_zone": "inland_monsoon" },
    "kerala": { "latitude": 10.8505, "longitude": 76.2711, "climate_zone": "coastal_monsoon" }
}

def load_and_clean_data(file_path: str, label_path: Optional[str] = None, city_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw epidemiological data and standardize temporal information.
    
    Args:
        file_path: Path to the raw CSV file
        label_path: Optional path to separate labels/outbreak indicators
        city_filter: Filter results to a specific city/region
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    date_col = 'week_start_date' if 'week_start_date' in df.columns else 'date'
    if date_col in df.columns:
        df['date'] = pd.to_datetime(df[date_col])
    else:
        if 'year' in df.columns and 'weekofyear' in df.columns:
            df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['weekofyear'].astype(str) + '-1', format='%Y-%W-%w')
        else:
            raise ValueError("No valid date column found.")

    if label_path:
        labels = pd.read_csv(label_path)
        merge_keys = ['city', 'year', 'weekofyear']
        if all(k in df.columns for k in merge_keys) and all(k in labels.columns for k in merge_keys):
            df = pd.merge(df, labels, on=merge_keys, how='left', suffixes=('', '_label'))
        else:
            logger.warning("Merge keys missing in label file. Skipping labels.")
    
    if city_filter and 'city' in df.columns:
        df = df[df['city'] == city_filter]
    
    df = df.sort_values('date').reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def impute_missing_values(df: pd.DataFrame, temp_col: str, precip_col: str, cases_col: str) -> pd.DataFrame:
    """
    Impute missing environmental and case data using simple, robust strategies.
    
    Args:
        df: Input dataframe with potential missing values
        temp_col: Temperature column name (interpolate small gaps)
        precip_col: Precipitation column name (forward-fill short runs)
        cases_col: Case count column (fill missing with 0)
    
    Returns:
        A dataframe with imputed environmental variables and case counts.
    """
    df = df.copy()

    # Temperature: short linear interpolation, then gentle back/forward fill
    if temp_col and temp_col in df.columns:
        df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
        df[temp_col] = df[temp_col].interpolate(limit=3).ffill().bfill()

    # Precipitation: forward-fill a few weeks, then backfill if needed
    if precip_col and precip_col in df.columns:
        df[precip_col] = pd.to_numeric(df[precip_col], errors='coerce')
        df[precip_col] = df[precip_col].ffill(limit=3).bfill(limit=3)

    # Cases: missing becomes interpolated (if it's set to 0, momentum and other features will tweak out)
    if cases_col in df.columns:
        df['missing_data'] = df[cases_col].isna().astype(int)
        df[cases_col] = df[cases_col].interpolate(method='linear', limit=2)
        df[cases_col] = np.maximum(df[cases_col].astype(int), 0)

    return df

def feature_engineering_momentum(df: pd.DataFrame, cases_col: str, short_w: int = 4, long_w: int = 12) -> pd.DataFrame:
    """
    Calculate outbreak momentum using short-vs-long EMAs and a 1-week lag.
    
    Args:
        df: Input dataframe with case counts
        cases_col: Column containing weekly case counts
        short_w: Window size for short-term EMA
        long_w: Window size for long-term EMA
    
    Returns:
        Dataframe with `momentum` and `momentum_lag1` features.
    """
    df = df.copy()
    if cases_col in df.columns:
        short_ema = df[cases_col].ewm(span=short_w, adjust=False).mean()
        long_ema = df[cases_col].ewm(span=long_w, adjust=False).mean()
        df['momentum'] = short_ema - long_ema
    # Lag helps catch the turn before the visible spike
    df['momentum_lag1'] = df['momentum'].shift(1).fillna(0)
    return df

def feature_engineering_lags(df: pd.DataFrame, target_cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Generates time-lagged features for biological delays."""
    df = df.copy()
    for col in target_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def feature_engineering_cumulative(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """Calculates rolling sum (e.g., total rainfall in last 4 weeks)."""
    df = df.copy()
    if col in df.columns:
        df[f'{col}_cumulative_{window}w'] = df[col].rolling(window=window).sum()
    return df

def run_pipeline(
    file_path: str,
    label_path: Optional[str] = None,
    cases_col: str = 'total_cases',
    temp_col: Optional[str] = 'reanalysis_avg_temp_k',
    precip_col: Optional[str] = 'precipitation_amt_mm',
    city_filter: Optional[str] = None,
    method: str = '2sigma'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full data processing pipeline for dengue outbreak prediction.
    
    Args:
        file_path: Path to raw features CSV
        label_path: Path to labels/outbreak indicators CSV
        cases_col: Column name for case counts
        temp_col: Column name for temperature data
        precip_col: Column name for precipitation data
        city_filter: Filter data to specific city/region
        method: Outbreak detection method ('2sigma' or 'farrington')
    
    Returns:
        Tuple of (processed dataframe, list of feature column names)
    """
    logger.info(f"Starting Pipeline | City: {city_filter} | Method: {method.upper()}")
    
    # 1. Load and Standardize
    df = load_and_clean_data(file_path, label_path, city_filter)
    
    # 2. Impute
    df = impute_missing_values(df, temp_col, precip_col, cases_col)
    
    # 3. Detect Outbreaks (Ground Truth for training)
    from src.outbreak_detection import detect_outbreak_signal
    df['spike'] = detect_outbreak_signal(df, cases_col=cases_col, window=52, sigma=2.0)
    # 2-sigma has gotten me out of so many tight spots lol
        
    # 4. Phase 2 Feature Engineering
    # Momentum + Momentum_Lag1
    df = feature_engineering_momentum(df, cases_col)
    
    # Weather Lags
    weather_cols = [c for c in [temp_col, precip_col] if c]
    df = feature_engineering_lags(df, weather_cols, lags=[1, 4, 8])
    
    # Cumulative Rain
    if precip_col:
        df = feature_engineering_cumulative(df, precip_col, window=4)
        if f'{precip_col}_cumulative_4w' in df.columns:
            df = df.rename(columns={f'{precip_col}_cumulative_4w': 'precip_cumulative_4w'})
    
    # Mapping standardized names to internal notebook/model names
    rename_map = {}
    if temp_col:
        rename_map[f'{temp_col}_lag4'] = 'temperature_lag4'
    if 'reanalysis_relative_humidity_percent' in df.columns:
        df = df.rename(columns={'reanalysis_relative_humidity_percent': 'humidity'})
            
    df = df.rename(columns=rename_map)
        
    # Seasonal features
    df['month'] = df['date'].dt.month
    if 'week_of_year' not in df.columns:
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        
    # Set the Target (Predicting if NEXT week is a spike)
    df['target'] = df['spike'].shift(-1).fillna(0).astype(int)
    
    # 5. Best performing Feature Set(for Phase 2/RandomForest )
    feature_cols = [
        'precip_cumulative_4w', 
        'temperature_lag4', 
        'humidity', 
        'month', 
        'momentum_lag1'
    ]
    
    # Drop rows with NaNs created by lags/rolling windows
    df = df.dropna().reset_index(drop=True)
    
    # Final check: Ensure all recovery features are present
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        logger.warning(f"Missing features in final dataframe: {missing}")
    else:
        logger.info(f"Pipeline Complete. Final feature set ready with {len(feature_cols)} features.")
    
    return df, [f for f in feature_cols if f in df.columns]

# --- API Compatibility Wrappers ---
def get_processed_data(file_path, **kwargs):
    return run_pipeline(file_path, **kwargs)

class PipelineConfig:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def to_dict(self): return self.__dict__