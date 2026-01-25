import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Any, Optional

# R integration imports
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import conversion, default_converter
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_outbreak_signal(
    df: pd.DataFrame, 
    cases_col: str = 'total_cases', 
    window: int = 52, 
    sigma: float = 2.0, 
    min_cases: int = 5
) -> pd.Series:
    """
    Detect outbreak spikes using rolling statistical threshold (2-Sigma method).
    
    Args:
        df: Input dataframe with case counts
        cases_col: Column name containing case counts
        window: Rolling window size for baseline calculation (weeks)
        sigma: Number of standard deviations from mean for threshold
        min_cases: Minimum case count to classify as spike
        
    Returns:
        Series of binary spike indicators
    """
    series = df[cases_col].copy()
    
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    threshold = rolling_mean + (sigma * rolling_std)
    is_spike = (series > threshold).astype(int).fillna(0)
    
    # Prune spikes where absolute case count is too low (dry-season noise)
    is_spike[series < min_cases] = 0
    
    return is_spike

class FarringtonFlexibleEngine:
    """
    Implementation of Farrington Flexible algorithm for surveillance outbreak detection.
    Uses R integration via rpy2 for statistical rigor.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Farrington engine with R bridge and configurable parameters.
        
        Args:
            params: Dictionary of algorithm parameters (baseline years, window size, alpha level, etc.)
        """
        # Default parameters based on epidemiological best practices
        self.params = params or {
            'b': 2,            # Years back for baseline
            'w': 1,            # Half-window size (weeks)
            'alpha': 0.10,     # Significance level 
            'trend': False,    # Disable trend to catch rising epidemics as outbreaks
            'reweight': True,  # Re-weight past outbreaks
            'min_cases': 5     # Minimum cases to call an outbreak
        }

        try:
            if not R_AVAILABLE:
                raise ImportError("rpy2 not installed")
                
            self.surveillance = importr('surveillance')
            self.r_bridge_active = True
            logger.info("R bridge initialized successfully")
            # if this stays up, my day gets instantly better
        except Exception as e:
            logger.error(f"Failed to initialize R bridge: {e}")
            self.r_bridge_active = False
        
    def get_labels(self, df: pd.DataFrame, cases_col: str = 'total_cases') -> pd.DataFrame:
        """
        Compute outbreak labels using Farrington Flexible algorithm.
        
        Args:
            df: Input dataframe with case counts and temporal information
            cases_col: Column name for case counts
        
        Returns:
            Dataframe with added `spike_farrington` column.
        """
        min_cases = self.params.get('min_cases', 5)
        
        # Fallback to 2-sigma if R is broken
        if not self.r_bridge_active:
            print(f"\n{'!'*50}")
            print("WARNING: R BRIDGE INACTIVE. FALLING BACK TO 2-SIGMA.")
            print(f"{'!'*50}\n")
            logger.warning("R bridge inactive. Falling back to 2-sigma detection.")
            # i rlly hope the r-bridge will work this time haha
            df = df.copy()
            df['spike_farrington'] = detect_outbreak_signal(
                df, cases_col=cases_col, min_cases=min_cases
            )
            return df
        
        print(f"\n{'='*50}")
        print("CONFIRMED: Using R-Surveillance Farrington Flexible Algorithm")
        print(f"{'='*50}\n")
        
        df = df.copy()
        n_rows = len(df)

        # Get params with defaults
        b = self.params.get('b', 3)
        w = self.params.get('w', 2)

        # Calculate minimum baseline needed
        min_baseline = b * 52 + 2 * w + 4
        monitor_start = min_baseline

        logger.info(
            "Farrington params | n_rows=%s b=%s w=%s min_baseline=%s monitor_start=%s",
            n_rows,
            b,
            w,
            min_baseline,
            monitor_start
        )
        
        if n_rows <= monitor_start:
            monitor_start = max(int(n_rows * 0.6), 52)
            b = max(1, (monitor_start - 2 * w - 4) // 52)
            logger.warning(
                "Dataset short. Adjusted: b=%s monitor_start=%s",
                b,
                monitor_start
            )
        
        try:
            with (conversion.localconverter(default_converter + pandas2ri.converter)):
                # 1. Prepare Data for R
                start_year = int(df['year'].iloc[0])
                start_week = int(df['week_of_year'].iloc[0]) if 'week_of_year' in df.columns else 1

                observed = ro.IntVector(df[cases_col].fillna(0).astype(int).values)
                sts_obj = self.surveillance.sts(
                    observed=observed,
                    frequency=52,
                    start=ro.IntVector([start_year, start_week])
                )

                # 2. Execute R Algorithm
                alpha_val = self.params.get('alpha', 0.1)
                
                control = ro.ListVector({
                    'range': ro.IntVector(range(monitor_start + 1, n_rows + 1)),
                    'b': ro.IntVector([b]),
                    'w': ro.IntVector([w]),
                    'alpha': ro.FloatVector([alpha_val]),
                    'trend': ro.BoolVector([self.params.get('trend', True)]),
                    'reweight': ro.BoolVector([self.params.get('reweight', True)])
                })
                
                try:
                    result = self.surveillance.farringtonFlexible(sts_obj, control=control)
                    alarms = np.array(result.slots['alarm']).astype(int).flatten()
                    final_result = result
                except Exception as e:
                    logger.error(f"R execution failed: {e}")
                    final_result = None

                # 3. Extract Alarms
                if final_result:
                    alarms = np.array(final_result.slots['alarm']).astype(int).flatten()
                else:
                    alarms = np.zeros(n_rows - monitor_start)

            # 4. Alignment
            spike_col = np.zeros(n_rows)
            end_idx = min(monitor_start + len(alarms), n_rows)
            spike_col[monitor_start:end_idx] = alarms[:end_idx - monitor_start]
            
            # 5. Consistent Backfill
            baseline_slice = df.iloc[:monitor_start]
            if len(baseline_slice) > 0:
                base_spikes = detect_outbreak_signal(baseline_slice, cases_col=cases_col, min_cases=min_cases)
                spike_col[:monitor_start] = base_spikes

        except Exception as e:
            logger.error(f"Farrington Flexible Execution Error: {e}")
            spike_col = np.zeros(n_rows)

        # Minimum Case Filtering
        spike_col[df[cases_col] < min_cases] = 0
        
        df['spike_farrington'] = spike_col.astype(int)
        return df