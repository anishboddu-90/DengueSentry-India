import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from rpy2.robjects import conversion, default_converter
from rpy2.robjects.vectors import DataFrame

with (conversion.localconverter(default_converter + pandas2ri.converter)):
    surveillance = importr('surveillance')
    def farrington_flexible_label(df, cases_col='total_cases'):
        start_year = int(df['year'].iloc[0])
        start_week = int(df['weekofyear'].iloc[0])
        
        observed = ro.IntVector(df[cases_col].values)
        
        # Create STS object
        sts_obj = surveillance.sts(observed=observed,frequency=52,start=ro.IntVector([start_year, start_week]))
        
        # Set Control
        monitor_start = 156 
        if len(df) <= monitor_start:
            monitor_start = len(df) // 2 # Fallback for very short data
            
        control = ro.ListVector({
            'range': ro.IntVector(range(monitor_start + 1, len(df) + 1)), # R is 1-indexed
            'b': 3,            # 3 years back
            'w': 1,            # Window of 1 week
            'alpha': 0.01,     # 99% confidence (more robust against noise)
            'trend': True,
            'noPeriods': 1,
            'reweight': True
        })

        
        
    # Run Algorithm with error handling
        try:
            result = surveillance.farringtonFlexible(sts_obj, control=control)
            alarms = np.array(result.slots['alarm']).astype(int).flatten()
        except Exception as e:
            print(f"FarringtonFlexible failed: {e}")
            print("Falling back to zero alarms")
            alarms = np.zeros(len(df) - monitor_start)

        # Ensure alarms length matches the monitored period
        monitored_length = len(df) - monitor_start
        if len(alarms) < monitored_length:
            # Pad with zeros if needed
            alarms = np.pad(alarms, (0, monitored_length - len(alarms)), 'constant')
        elif len(alarms) > monitored_length:
            alarms = alarms[:monitored_length]

        # Map back to df
        spike_col = np.zeros(len(df))
        spike_col[monitor_start:] = alarms
        df['spike_farrington'] = spike_col.astype(int)
        
        return df