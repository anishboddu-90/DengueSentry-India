import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import warnings
import logging
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

warnings.filterwarnings("ignore")

# Setup logging for pipeline execution tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class ValidationConfig:
    """
    Configuration for data validation thresholds.
    
    Defines acceptable ranges for epidemiological and meteorological data
    to catch data quality issues before modeling.
    """
    min_cases: int = 0                # Minimum acceptable case count (cannot be negative)
    max_cases: int = 10000            # Maximum plausible case count for a city
    min_temp_c: float = -10.0         # Minimum temperature for dengue-prone regions
    max_temp_c: float = 50.0          # Maximum temperature for dengue-prone regions
    min_precip_mm: float = 0.0        # Minimum precipitation (cannot be negative)
    max_precip_mm: float = 1000.0     # Maximum weekly precipitation (extreme event threshold)
    max_date_gap_days: int = 21       # Maximum acceptable gap between data points
    max_temp_change_c: float = 10.0   # Maximum week-to-week temperature change (sensor error check)
    max_identical_weeks: int = 8      # Maximum consecutive weeks with identical case counts

@dataclass
class PipelineConfig:
    """
    Main configuration for the dengue data processing pipeline.
    
    Contains all parameters needed for data loading, cleaning, feature engineering,
    and outbreak detection.
    """
    file_path: str                    # Path to features CSV file
    label_path: Optional[str] = None  # Path to labels CSV file (optional)
    
    cases_col: str = 'total_cases'    # Column name for dengue case counts
    temp_col: Optional[str] = None    # Column name for temperature data
    precip_col: Optional[str] = None  # Column name for precipitation data
    
    city_filter: Optional[str] = None # Filter data for specific city (e.g., 'sj', 'iq')
    
    spike_method: str = 'farrington'  # Outbreak detection method: 'farrington' or '2sigma'
    sigma: float = 2.0                # Standard deviation multiplier for 2σ method
    
    farrington_params: Dict[str, Any] = field(default_factory=lambda: {
        'b': 3,            # Years back for baseline calculation
        'w': 1,            # Half-window size for reference values
        'alpha': 0.01,     # Significance level (99% confidence interval)
        'trend': True,     # Adjust for long-term trend
        'noPeriods': 1,    # Number of seasonal periods to consider
        'reweight': True   # Re-weight past outbreaks to reduce influence
    })
    
    max_temp_gap_weeks: int = 2       # Maximum weeks to interpolate temperature gaps
    max_precip_gap_weeks: int = 1     # Maximum weeks to forward-fill precipitation gaps
    
    def to_dict(self) -> Dict:
        """Serialize configuration for experiment tracking."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# ============================================================================
# VALIDATION COMPONENT
# ============================================================================

class DataValidator:
    """
    Validates data quality for epidemiological time-series analysis.
    
    Performs checks for missing values, date continuity, physical plausibility,
    and outlier detection to ensure data integrity before modeling.
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(self, df: pd.DataFrame, 
                cases_col: str = 'total_cases',
                temp_col: str = None,
                precip_col: str = None) -> Dict[str, Any]:
        """
        Run comprehensive data validation checks.
        
        Returns:
            Dictionary with validation results including critical issues,
            warnings, and summary statistics.
        """
        
        issues = {'critical': [], 'warning': [], 'info': []}
        
        # Run validation checks in sequence
        issues = self._validate_dates(df, issues)
        issues = self._validate_cases(df, cases_col, issues)
        
        if temp_col:
            issues = self._validate_temperature(df, temp_col, issues)
        
        if precip_col:
            issues = self._validate_precipitation(df, precip_col, issues)
        
        return {
            'has_critical': len(issues['critical']) > 0,
            'has_warnings': len(issues['warning']) > 0,
            'critical_issues': issues['critical'],
            'warning_issues': issues['warning'],
            'info_issues': issues['info'],
            'summary': self._generate_summary(issues)
        }
    
    def _validate_dates(self, df: pd.DataFrame, issues: Dict) -> Dict:
        """Validate date continuity and consistency."""
        if 'date' not in df.columns:
            issues['critical'].append("Missing date column")
            return issues
        
        df_sorted = df.sort_values('date')
        date_diff = df_sorted['date'].diff().dt.days
        
        # Check for large gaps in time series
        large_gaps = date_diff[date_diff > self.config.max_date_gap_days]
        if not large_gaps.empty:
            issues['warning'].append(f"Date gaps >{self.config.max_date_gap_days} days: {len(large_gaps)}")
        
        # Check for duplicate dates
        duplicate_dates = df_sorted['date'].duplicated().sum()
        if duplicate_dates > 0:
            issues['critical'].append(f"Duplicate dates: {duplicate_dates}")
        
        # Check data spans sufficient timeframe for analysis
        date_range = df_sorted['date'].max() - df_sorted['date'].min()
        if date_range.days < 365:
            issues['info'].append(f"Short time span: {date_range.days} days")
        
        return issues
    
    def _validate_cases(self, df: pd.DataFrame, cases_col: str, issues: Dict) -> Dict:
        """Validate dengue case count data."""
        if cases_col not in df.columns:
            issues['critical'].append(f"Missing cases column: {cases_col}")
            return issues
        
        cases = df[cases_col]
        missing_cases = cases.isnull().sum()
        if missing_cases > 0:
            issues['warning'].append(f"Missing cases: {missing_cases} rows")
        
        # Check for physically impossible values
        negative_cases = (cases < self.config.min_cases).sum()
        if negative_cases > 0:
            issues['critical'].append(f"Negative cases: {negative_cases} rows")
        
        # Check for extreme values (potential data entry errors)
        high_cases = (cases > self.config.max_cases).sum()
        if high_cases > 0:
            issues['warning'].append(f"Extreme cases >{self.config.max_cases}: {high_cases} rows")
        
        # Check for suspicious patterns
        if (cases == 0).all():
            issues['critical'].append("All case values are zero")
        elif cases.nunique() == 1 and len(cases) > 10:
            issues['warning'].append(f"Constant case values: {cases.iloc[0]}")
        
        # Check for long streaks of identical values (reporting system stuck)
        if len(cases) > 20:
            max_streak = self._max_consecutive_equal(cases)
            if max_streak > self.config.max_identical_weeks:
                issues['warning'].append(f"Long identical case streak: {max_streak} weeks")
        
        return issues
    
    def _validate_temperature(self, df: pd.DataFrame, temp_col: str, issues: Dict) -> Dict:
        """Validate temperature data quality."""
        if temp_col not in df.columns:
            issues['warning'].append(f"Missing temperature column: {temp_col}")
            return issues
        
        temp = df[temp_col]
        missing_temp = temp.isnull().sum()
        if missing_temp > 0:
            issues['warning'].append(f"Missing temperature: {missing_temp} rows")
        
        # Check temperature within physically plausible range for dengue transmission
        low_temp = (temp < self.config.min_temp_c).sum()
        if low_temp > 0:
            issues['critical'].append(f"Temp <{self.config.min_temp_c}°C: {low_temp} rows")
        
        high_temp = (temp > self.config.max_temp_c).sum()
        if high_temp > 0:
            issues['critical'].append(f"Temp >{self.config.max_temp_c}°C: {high_temp} rows")
        
        # Check for rapid temperature changes (sensor errors)
        if len(temp) > 10:
            temp_change = temp.diff().abs()
            large_changes = temp_change[temp_change > self.config.max_temp_change_c].count()
            if large_changes > 2:
                issues['warning'].append(f"Large temp changes >{self.config.max_temp_change_c}°C: {large_changes}")
        
        return issues
    
    def _validate_precipitation(self, df: pd.DataFrame, precip_col: str, issues: Dict) -> Dict:
        """Validate precipitation data quality."""
        if precip_col not in df.columns:
            issues['warning'].append(f"Missing precipitation column: {precip_col}")
            return issues
        
        precip = df[precip_col]
        missing_precip = precip.isnull().sum()
        if missing_precip > 0:
            issues['warning'].append(f"Missing precipitation: {missing_precip} rows")
        
        # Precipitation cannot be negative
        negative_precip = (precip < self.config.min_precip_mm).sum()
        if negative_precip > 0:
            issues['critical'].append(f"Negative precipitation: {negative_precip} rows")
        
        # Check for extreme precipitation events
        high_precip = (precip > self.config.max_precip_mm).sum()
        if high_precip > 0:
            issues['warning'].append(f"Extreme precipitation >{self.config.max_precip_mm}mm: {high_precip} rows")
        
        return issues
    
    def _max_consecutive_equal(self, series: pd.Series) -> int:
        """Calculate maximum consecutive equal values in a time series."""
        max_streak = 0
        current_streak = 1
        
        for i in range(1, len(series)):
            if series.iloc[i] == series.iloc[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        return max_streak
    
    def _generate_summary(self, issues: Dict) -> str:
        """Generate human-readable validation summary."""
        critical_count = len(issues['critical'])
        warning_count = len(issues['warning'])
        info_count = len(issues['info'])
        
        if critical_count > 0:
            return f"FAILED: {critical_count} critical issues"
        elif warning_count > 0:
            return f"WARNINGS: {warning_count} issues to review"
        elif info_count > 0:
            return f"OK: {info_count} notes"
        else:
            return "PASSED: No issues found"

# ============================================================================
# PIPELINE COMPONENTS
# ============================================================================

class MissingValueHandler:
    """
    Handles missing values with domain-appropriate imputation strategies.
    
    Uses different methods for different data types:
    - Temperature: Linear interpolation (continuous measurement)
    - Precipitation: Forward fill (rainfall patterns persist)
    - Cases: Zero fill (missing report = no cases)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply appropriate imputation for each column type."""
        df = df.copy()
        
        # Temperature: linear interpolation for continuous measurement
        if self.config.temp_col and self.config.temp_col in df.columns:
            df[self.config.temp_col] = df[self.config.temp_col].interpolate(
                method='linear',
                limit=self.config.max_temp_gap_weeks
            )
        
        # Precipitation: forward fill (rainfall patterns often persist)
        if self.config.precip_col and self.config.precip_col in df.columns:
            df[self.config.precip_col] = df[self.config.precip_col].ffill(
                limit=self.config.max_precip_gap_weeks
            )
        
        # Cases: assume missing report means no cases
        if self.config.cases_col in df.columns:
            df[self.config.cases_col] = df[self.config.cases_col].fillna(0)
        
        return df

class SpikeDetector:
    """
    Detects outbreak spikes using statistical methods.
    
    Implements two methods:
    1. Farrington Flexible Algorithm (gold standard for outbreak detection)
    2. 2-Sigma method (simple statistical threshold)
    """
    
    def __init__(self, method="farrington", sigma: float = 2.0, farrington_params: Optional[Dict] = None):
        self.method = method
        self.sigma = sigma
        self.farrington_params = farrington_params if farrington_params else {}

    def detect(self, df: pd.DataFrame, cases_col: str) -> pd.Series:
        """Detect outbreak spikes in case time series."""
        if self.method == "farrington":
            return self._farrington(df, cases_col)
        elif self.method == "2sigma":
            return self._two_sigma(df, cases_col)
        else:
            raise ValueError(f"Unknown spike detection method: {self.method}")
    
    def _two_sigma(self, df: pd.DataFrame, cases_col: str) -> pd.Series:
        mean_52 = df[cases_col].rolling(window=52, min_periods=1).mean()
        std_52 = df[cases_col].rolling(window=52, min_periods=1).std()
        spikes = ((df[cases_col] - mean_52) > self.sigma * std_52).astype(int)
        return spikes
    
    def _farrington(self, df: pd.DataFrame, cases_col: str) -> pd.Series:
        """
        Farrington Flexible Algorithm for outbreak detection.
        
        Uses R's surveillance package via rpy2 bridge for gold-standard
        outbreak detection with seasonality and trend adjustments.
        """
        try:
            from src.outbreak_detection import farrington_flexible_label
            result_df = farrington_flexible_label(
                df.copy(), 
                cases_col=cases_col,
                **self.farrington_params
            )
            return result_df['spike_farrington']
        except ImportError as e:
            logger.error(f"Failed to import Farrington module: {e}")
            logger.warning(f"Falling back to 2-sigma method with sigma={self.sigma}")
            return self._two_sigma(df, cases_col)
        except Exception as e:
            logger.error(f"Farrington algorithm failed: {e}")
            logger.warning(f"Falling back to 2-sigma method with sigma={self.sigma}")
            return self._two_sigma(df, cases_col)

class FeatureEngineer:
    """
    Creates predictive features from raw epidemiological and meteorological data.
    
    Generates:
    1. Case momentum (short-term vs long-term trends)
    2. Weather lag features (1, 4, 8 week lags)
    3. Cumulative precipitation
    4. Seasonal features (week of year, month)
    """
    
    def __init__(self, temp_col: Optional[str] = None, precip_col: Optional[str] = None):
        self.temp_col = temp_col
        self.precip_col = precip_col

    def create(self, df: pd.DataFrame, cases_col: str) -> Tuple[pd.DataFrame, List[str]]:
        """Create all predictive features from raw data."""
        df = df.copy()
        
        # Check if columns already exist before creating them
        features = []
        
        # 1. Case momentum
        if 'momentum' not in df.columns:
            df['momentum'] = self._calculate_momentum(df[cases_col])
        features.append('momentum')
        
        # 2. Temperature lag features
        if self.temp_col and self.temp_col in df.columns:
            for lag in [1, 4, 8]:
                col_name = f'temp_lag{lag}'
                if col_name not in df.columns:
                    df[col_name] = df[self.temp_col].shift(lag)
                features.append(col_name)
        
        # 3. Precipitation lag features
        if self.precip_col and self.precip_col in df.columns:
            for lag in [1, 4, 8]:
                col_name = f'precip_lag{lag}'
                if col_name not in df.columns:
                    df[col_name] = df[self.precip_col].shift(lag)
                features.append(col_name)
            
            cum_col = 'precip_cumulative_4w'
            if cum_col not in df.columns:
                df[cum_col] = df[self.precip_col].rolling(window=4).sum()
            features.append(cum_col)
        
        # 4. Seasonal features (only add if not already present)
        if 'week_of_year' not in df.columns:
            df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        if 'week_of_year' not in features:
            features.append('week_of_year')
            
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'month' not in features:
            features.append('month')
        
        return df, features
    
    def _calculate_momentum(self, cases: pd.Series) -> pd.Series:
        """
        Calculate case momentum as difference between short and long-term EMAs.
        
        Short EMA (4 weeks): Recent trend
        Long EMA (12 weeks): Baseline trend
        Difference: Acceleration/deceleration of cases
        """
        short_ema = cases.ewm(span=4, adjust=False).mean()   # 1-month trend
        long_ema = cases.ewm(span=12, adjust=False).mean()   # 3-month baseline
        return short_ema - long_ema

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class DengueDataPipeline:
    """
    Main data preprocessing pipeline for dengue outbreak prediction.
    
    Controls all components:
    1. Data loading and merging
    2. Data validation
    3. Missing value handling
    4. Outbreak spike detection
    5. Feature engineering
    6. Final cleaning and preparation
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.raw_df = None
        self.processed_df = None
        self.features = None
        
        # Initialize metadata for experiment tracking
        self.metadata = {
            'config': config.to_dict(),
            'pipeline_version': '1.0',
            'run_timestamp': datetime.now().isoformat()
        }

        # Create ValidationConfig
        self.validator_config = ValidationConfig(
            min_cases=0,
            max_cases=10000,
            min_temp_c=-10.0,
            max_temp_c=50.0,
            min_precip_mm=0.0,
            max_precip_mm=1000.0,
            max_date_gap_days=21,
            max_temp_change_c=10.0,
            max_identical_weeks=8
        )
        
        # Initialize pipeline components
        self.validator = DataValidator(self.validator_config)
        self.missing_handler = MissingValueHandler(config)
        self.spike_detector = SpikeDetector(
            method=config.spike_method,
            sigma=config.sigma,
            farrington_params=config.farrington_params
        )
        self.feature_engineer = FeatureEngineer(
            temp_col=config.temp_col,
            precip_col=config.precip_col
        )
    
    def run(self, validate: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute the complete data preprocessing pipeline.
        
        Parameters:
            validate (bool): Whether to perform data validation checks
        
        Returns:
            processed_df: Dataframe ready for modeling
            features: List of feature column names
        """
        logger.info("Starting dengue data preprocessing pipeline")

        try:
            # 1. Load and prepare raw data
            self.raw_df = self._load_raw_data()
            self.metadata['raw_rows'] = len(self.raw_df)
            logger.info(f"Loaded {len(self.raw_df)} rows of raw data")

            # 2. Validate data quality (optional)
            if validate:
                self._validate_data()
            
            # 3. Handle missing values with domain-appropriate methods
            self.raw_df = self.missing_handler.handle(self.raw_df)
            
            # 4. Detect outbreak spikes using selected method
            spikes = self.spike_detector.detect(self.raw_df, self.config.cases_col)
            self.raw_df['spike'] = spikes.astype(int)
            self.metadata['spike_rate'] = spikes.mean()
            
            # 5. Create prediction target (next week's outbreak)
            self.raw_df['target'] = self.raw_df['spike'].shift(-1)

            # 6. Engineer predictive features
            self.processed_df, self.features = self.feature_engineer.create(
                self.raw_df,
                self.config.cases_col
            )

            # 7. Final cleaning and preparation
            self._final_cleanup()
            self.metadata['processed_rows'] = len(self.processed_df)
            self.metadata['features'] = self.features

            # 8. Log comprehensive results
            self._log_results()
            logger.info("Pipeline completed successfully!")
            
            return self.processed_df, self.features
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load and prepare raw data with proper date handling and merging."""
        df = pd.read_csv(self.config.file_path)
        
        # Standardize date column across different dataset formats
        date_col = 'week_start_date' if 'week_start_date' in df.columns else 'date'
        df['date'] = pd.to_datetime(df[date_col])
        df['year'] = df['date'].dt.year
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Merge with labels if provided
        if self.config.label_path:
            labels = pd.read_csv(self.config.label_path)
            
            # Ensure labels have same date structure
            if 'date' not in labels.columns:
                # Check for different date column names
                for possible_date_col in ['week_start_date', 'date', 'week_end_date']:
                    if possible_date_col in labels.columns:
                        labels['date'] = pd.to_datetime(labels[possible_date_col])
                        break
            
            # If still no date, try to construct from year/week
            if 'date' not in labels.columns:
                if 'year' in labels.columns and 'weekofyear' in labels.columns:
                    labels['date'] = pd.to_datetime(
                        labels['year'].astype(str) + '-' + 
                        labels['weekofyear'].astype(str) + '-1', 
                        format='%Y-%W-%w'
                    )
                else:
                    raise ValueError("Cannot determine dates from labels file")
            
            # Add year and week columns to labels
            labels['year'] = labels['date'].dt.year
            labels['week_of_year'] = labels['date'].dt.isocalendar().week.astype(int)
            
            # FIX 3: Identify overlapping columns BEFORE merge
            overlapping_cols = set(df.columns).intersection(set(labels.columns))
            overlapping_cols.discard('city')
            overlapping_cols.discard('year')
            overlapping_cols.discard('week_of_year')
            overlapping_cols.discard('date')
            
            if overlapping_cols:
                logger.warning(f"Overlapping columns will be renamed: {overlapping_cols}")
                # Rename overlapping columns in labels before merge
                rename_dict = {col: f"{col}_label" for col in overlapping_cols}
                labels = labels.rename(columns=rename_dict)
            
            # FIX 4: Use explicit merge columns and suffix strategy
            merge_cols = ['city', 'year', 'week_of_year']
            df = pd.merge(
                df, 
                labels, 
                on=merge_cols, 
                how='left',
                suffixes=('', '_label')
            )
        
        # Filter for specific city if requested
        if self.config.city_filter and 'city' in df.columns:
            df = df[df['city'] == self.config.city_filter].copy()
        
        # Sort chronologically for time-series analysis
        df = df.sort_values('date').reset_index(drop=True)
        
        # FIX 5: Remove any duplicate columns that might have slipped through
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def _validate_data(self) -> bool:
        """Run comprehensive data validation and log results."""
        logger.info("Running data validation checks")
        
        validation_result = self.validator.validate(
            self.raw_df,
            cases_col=self.config.cases_col,
            temp_col=self.config.temp_col,
            precip_col=self.config.precip_col
        )
        
        self.metadata['validation'] = validation_result
        
        if validation_result['has_critical']:
            logger.warning(f"Critical validation issues: {validation_result['critical_issues']}")
        
        if validation_result['has_warnings']:
            logger.info(f"Validation warnings: {validation_result['warning_issues']}")
        
        # Always return True to continue pipeline (research data is often messy)
        return True
    
    def _final_cleanup(self):
        """Final data cleaning and preparation for modeling."""
        # Keep only necessary columns
        keep_cols = (self.features + 
                    ['target', 'date', self.config.cases_col, 'spike', 'week_of_year', 'month'])
        
        # Remove rows with missing features or target
        self.processed_df = self.processed_df.dropna(
            subset=self.features + ['target']
        )[keep_cols].reset_index(drop=True)

        # Ensure proper data types for modeling
        self.processed_df['spike'] = self.processed_df['spike'].astype(int)
        self.processed_df['target'] = self.processed_df['target'].astype(int)
    
    def _log_results(self):
        """Log comprehensive pipeline results for documentation."""
        spike_rate = self.metadata.get('spike_rate', 0)

        logger.info("=" * 60)
        logger.info("DENGUE DATA PIPELINE - EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Detection Method:    {self.config.spike_method.upper()}")
        logger.info(f"City:                {self.config.city_filter or 'All Cities'}")
        logger.info(f"Raw Data Rows:       {self.metadata.get('raw_rows', 0):,}")
        logger.info(f"Processed Rows:      {self.metadata.get('processed_rows', 0):,}")
        logger.info(f"Outbreak Rate:       {spike_rate:.2%}")
        logger.info(f"Features Generated:  {len(self.features)}")
        logger.info("-" * 60)
        logger.info("Feature Categories:")
        logger.info(f"  • Case Trend:      {[f for f in self.features if 'momentum' in f]}")
        logger.info(f"  • Weather Lags:    {[f for f in self.features if 'temp' in f or 'precip' in f]}")
        logger.info(f"  • Seasonality:     week_of_year, month")
        logger.info("=" * 60)
    
    def save_metadata(self, path: str):
        """
        Save pipeline metadata for experiment reproducibility.
        
        Parameters:
            path: File path to save metadata JSON
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        metadata = {
            **self.metadata,
            'config': self.config.to_dict(),
            'feature_count': len(self.features),
            'features': self.features,
            'processed_shape': self.processed_df.shape if self.processed_df is not None else None
        }

        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Pipeline metadata saved to {path}")
    
    def save_processed_data(self, path: str):
        """
        Save processed data to CSV file.
        
        Parameters:
            path: File path to save processed data
        """
        if self.processed_df is None:
            raise ValueError("Processed data not available. Run pipeline first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.processed_df.to_csv(path, index=False)
        logger.info(f"Processed data saved to {path}")

# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def get_processed_data(
    file_path: str,
    label_path: Optional[str] = None,
    cases_col: str = 'total_cases',
    temp_col: Optional[str] = None,
    precip_col: Optional[str] = None,
    city_filter: Optional[str] = None,
    method: str = '2sigma',
    sigma: float = 2.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Legacy function for backward compatibility.
    Uses the new pipeline internally for cleaner code organization.
    """
    
    config = PipelineConfig(
        file_path=file_path,
        label_path=label_path,
        cases_col=cases_col,
        temp_col=temp_col,
        precip_col=precip_col,
        city_filter=city_filter,
        spike_method=method,
        sigma=sigma
    )
    
    pipeline = DengueDataPipeline(config)
    return pipeline.run()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the pipeline functionality.
    """
    
    # Example configuration for San Juan data
    config = PipelineConfig(
        file_path="../data/dengue_features_train.csv",
        label_path="../data/dengue_labels_train.csv",
        cases_col='total_cases',
        temp_col='station_avg_temp_c',
        precip_col='precipitation_amt_mm',
        city_filter='sj',
        spike_method='farrington',
        sigma=2.0
    )
    
    # Execute pipeline
    pipeline = DengueDataPipeline(config)
    df, features = pipeline.run()
    
    # Save results for reproducibility
    pipeline.save_metadata("outputs/experiment_1_metadata.json")
    pipeline.save_processed_data("outputs/processed_data.csv")
    
    # Display sample results
    logger.info("\nData Sample (First 5 Rows):")
    logger.info(df[['date', 'total_cases', 'spike', 'target'] + features[:3]].head())