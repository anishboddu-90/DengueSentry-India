import requests
import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_nasa_weather_data(lat: float, lon: float, start_date: datetime, end_date: datetime, cache_dir: str = "cache/weather", region_name: str = "region") -> pd.DataFrame:
    """
    Retrieves and processes meteorological data from NASA POWER using a functional approach.
    Handles API requests, caching, and data formatting.
    
    Args:
        lat: Latitude of location
        lon: Longitude of location
        start_date: Start date for query
        end_date: End date for query
        cache_dir: Local directory to store/retrieve results
        region_name: Identifier for the region (used in filename)
        
    Returns:
        DataFrame containing weekly aggregated weather data.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{region_name}_weather.csv")
    
    # Check cache first so we don't ping NASA unnecessarily
    # i don't get banned for spamming requests lol
    if os.path.exists(cache_path):
        logger.info(f"Loading weather data from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df

    # Configure API Request
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    community = "AG" # Agroclimatology community usually has the best data
    
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M",
        "community": community,
        "longitude": lon,
        "latitude": lat,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    
    try:
        logger.info(f"Fetching fresh data from NASA POWER for {region_name}...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Process JSON response into DataFrame
        df_daily = pd.DataFrame(data['properties']['parameter'])
        df_daily.index = pd.to_datetime(df_daily.index)
        df_daily = df_daily.reset_index().rename(columns={'index': 'date'})
        
        # Keeping units consistent here saves headaches later
        df_weekly = aggregate_daily_to_weekly(df_daily)
        
        # Cache results
        df_weekly.to_csv(cache_path, index=False)
        logger.info(f"Successfully cached weather data to {cache_path}")
        
        return df_weekly
        
    except Exception as e:
        logger.error(f"NASA POWER API failed: {e}. Probably my internet or nasa is down.")
        return pd.DataFrame()

def aggregate_daily_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily meteorological measurements into weekly epidemiological periods.
    
    Uses domain-appropriate aggregation: mean for continuous variables (temperature, humidity)
    and sum for accumulation variables (precipitation).
    """
    
    agg_rules = {
        'T2M': 'mean',
        'T2M_MAX': 'max',
        'T2M_MIN': 'min',
        'PRECTOTCORR': 'sum', # Rain accumulates over the week
        'RH2M': 'mean',
        'ALLSKY_SFC_SW_DWN': 'mean',
        'WS2M': 'mean'
    }
    
    # Resample to Monday-start weeks to match epi data
    df_weekly = df_daily.resample('W-MON', on='date', label='left', closed='left').agg(agg_rules).reset_index()
    
    return df_weekly.rename(columns={
        'T2M': 'temp_mean',
        'T2M_MAX': 'temp_max',
        'T2M_MIN': 'temp_min',
        'PRECTOTCORR': 'precip_sum',
        'RH2M': 'humidity_mean',
        'ALLSKY_SFC_SW_DWN': 'solar_radiation',
        'WS2M': 'wind_speed'
    })

class WeatherFetcher:
    """
    Legacy wrapper for backward compatibility with object-oriented weather data access.
    """
    def __init__(self, cache_dir: str = "cache/weather"):
        """
        Initialize weather fetcher with optional cache directory.
        
        Args:
            cache_dir: Directory for storing cached weather data
        """
        self.cache_dir = cache_dir

    def fetch_weather(
        self, 
        lat: float, 
        lon: float, 
        start_date: datetime, 
        end_date: datetime,
        region_name: str
    ) -> pd.DataFrame:
        """
        Fetch and aggregate weather data for specified location and period.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            start_date: Query start date
            end_date: Query end date
            region_name: Name identifier for caching
            
        Returns:
            Weekly aggregated weather dataframe
        """
        return fetch_nasa_weather_data(lat, lon, start_date, end_date, self.cache_dir, region_name)
    
    def _aggregate_to_weekly(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Delegate to functional aggregation implementation."""
        return aggregate_daily_to_weekly(df_daily)