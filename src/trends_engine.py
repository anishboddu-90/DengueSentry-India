import pandas as pd
import logging
from pytrends.request import TrendReq
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_dengue_trends(
    start_date: str,
    end_date: str,
    keywords: list = ["dengue", "fever", "mosquito", "platelets", "dengue symptoms"],
    geo: str = "IN-KL" # defaults to Kerala
) -> pd.DataFrame:
    """
    Fetch Google Trends data for dengue-related search keywords.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        keywords: List of search terms to track
        geo: Geographic region code (e.g., 'IN-KL' for Kerala)
        
    Returns:
        Dataframe with weekly aggregated trend indices
    """
    try:
        pytrends = TrendReq(hl='en-US', tz=330)
        
        # Build payload
        pytrends.build_payload(
            kw_list=keywords,
            cat=0,
            timeframe=f'{start_date} {end_date}',
            geo=geo,
            gprop=''
        )
        
        data = pytrends.interest_over_time()
        
        if data.empty:
            logger.warning("Google Trends returned no data.")
            # Sometimes the API just shrugs and gives us nothing lol
            return pd.DataFrame()
            
        data = data.reset_index()
        
        # Resample to weekly mean to match epidemiological data
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            # Align to Monday-start weeks to keep everything in sync
            # ALIGNMENT FIX: Resample to W-MON using label='left' (plays nice with case/weather)
            data = data.set_index('date').resample('W-MON', label='left', closed='left').mean().reset_index()
            
        # Metadata cleanup
        if 'isPartial' in data.columns:
            del data['isPartial']
            
        logger.info(f"Fetched {len(data)} weeks of trends data.")
        return data
        
    except Exception as e:
        logger.error(f"Failed to fetch Google Trends data: {e}")
        return pd.DataFrame()