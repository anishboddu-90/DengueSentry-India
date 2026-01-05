import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Any, Optional

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion, default_converter

logger = logging.getLogger(__name__)

class FarringtonFlexibleEngine:
    """
    Engine for executing Farrington Flexible outbreak detection algorithm.
    
    Handles:
    1. Python-R data conversion
    2. Algorithm execution with proper error handling  
    3. Result alignment and validation
    
    Parameters:
    -----------
    params : Optional[Dict[str, Any]]
        Algorithm parameters. Defaults to best epidemiological practices.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Farrington engine.
        
        Args:
            params: Algorithm parameters for Farrington Flexible
        """
        self.params = params or {
            'b': 3,            # Years back for baseline calculation
            'w': 2,            # Half-window size
            'alpha': 0.01,     # Significance level
            'trend': True,     # Adjust for long-term trend
            'noPeriods': 1,    # Number of seasonal periods to consider
            'reweight': True   # Re-weight past outbreaks to reduce influence
        }

        try:
            self.surveillance = importr('surveillance')
            self.r_bridge_active = True
            logger.info("R bridge created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize R bridge: {e}")
            self.r_bridge_active = False

    def get_labels(self, df: pd.DataFrame, cases_col: str = 'total_cases') -> pd.DataFrame:
        """
        Compute outbreak labels using Farrington Flexible algorithm.
        
        Args:
            df: Input dataframe with epidemiological time series
            cases_col: Column name with integer case counts
            
        Returns:
            DataFrame with added 'spike_farrington' column (0/1 labels)
            
        Raises:
            ValueError: If R bridge is inactive or cases_col is not found
        """
        if not self.r_bridge_active:
            logger.error("R bridge is not active")
            raise ValueError("R bridge inactive, check R installation.")
        
        df = df.copy()
        n_rows = len(df)

        # Determine start of monitoring period (usually 3 years/156 weeks)
        weeks_per_year = 52
        baseline_years = 3
        monitor_start = weeks_per_year * baseline_years # 156 weeks
        
        if n_rows <= monitor_start:
            monitor_start = n_rows // 2
            logger.info(f"Dataset is short: Adjusting start to {monitor_start}.")
        
        try:
            with (conversion.localconverter(default_converter + pandas2ri.converter)):
                # 1. Prepare STS Object
                start_year = int(df['year'].iloc[0])
                # Check multiple common column names for the starting week
                week_cols = ['week_of_year', 'weekofyear', 'week', 'epi_week']
                start_week = 1 # Default fallback

                for col in week_cols:
                    if col in df.columns:
                        start_week = int(df[col].iloc[0])
                        break

                observed = ro.IntVector(df[cases_col].fillna(0).astype(int).values)
                sts_obj = self.surveillance.sts(
                    observed=observed,
                    frequency=52,
                    start=ro.IntVector([start_year, start_week])
                )

                # 2. Configure Algorithm Control
                # R IS 1-INDEXED -> Add one to range
                control = ro.ListVector({
                    'range': ro.IntVector(range(monitor_start + 1, n_rows + 1)),
                    **self.params
                })

                # 3. Execute R Algorithm
                result = self.surveillance.farringtonFlexible(sts_obj, control=control)

                # 4. Extract Alarms
                alarms = np.array(result.slots['alarm']).astype(int).flatten()
            
            # 5. Result Alignment
            # The algorithm only returns results for the specified monitoring range
            spike_col = np.zeros(n_rows)

            actual_alarm_len = len(alarms)
            
            # We anchor the alarms to start exactly at monitor_start
            end_idx = min(monitor_start + actual_alarm_len, n_rows)
            actual_fill_len = end_idx - monitor_start

            if actual_alarm_len != (n_rows - monitor_start):
                logger.warning(f"Alarm length mismatch: Expected {n_rows - monitor_start}, got {actual_alarm_len}")

            # Correctly map the alarms starting from the monitor_start index
            spike_col[monitor_start:end_idx] = alarms[:actual_fill_len]

        except Exception as e:
            logger.error(f"Farrington Flexible Execution Error: {e}")
            spike_col = np.zeros(n_rows)

        df['spike_farrington'] = spike_col.astype(int)
        return df