import pandas as pd
import numpy as np
import logging
import warnings
from typing import Optional, Tuple, Dict

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DengueDataExtender:
    """
    Extend short dengue time series datasets to meet algorithm requirements.
    
    Many outbreak detection algorithms (like Farrington Flexible) require 5+ years of 
    historical data to accurately estimate seasonal baselines. This class synthesizes 
    plausible past data by learning patterns from available real data, then projecting 
    them backward temporally.
    """

    def __init__(self, real_data: pd.DataFrame, seed: int = 42):
        """
        Initialize the data extender with observed epidemic data.
        
        Args:
            real_data: DataFrame containing real case counts and temporal columns
            seed: Random seed for reproducibility
        """
        self.real_data = real_data.copy()
        self.seed = seed
        np.random.seed(seed)

        self._prepare_data()
        self.patterns = self._learn_patterns()

        print(f"Loaded {len(self.real_data)} weeks of real data")

    def _prepare_data(self):
        """
        Prepare input data: ensure temporal columns exist and detect data quality issues.
        """
        # Ensure date is datetime and sorted
        if 'date' in self.real_data.columns:
            self.real_data['date'] = pd.to_datetime(self.real_data['date'])
            self.real_data = self.real_data.sort_values('date').reset_index(drop=True)

        self.real_data['week_index'] = np.arange(len(self.real_data))

        # Extract temporal columns (year is REQUIRED for Farrington/Seasonality!!!!)
        self.real_data['year'] = self.real_data['date'].dt.year
        self.real_data['month'] = self.real_data['date'].dt.month
        self.real_data['week_of_year'] = self.real_data['date'].dt.isocalendar().week.astype(int)

        # Mark real data
        self.real_data['is_synthetic'] = False

        # Detect and log data issues for the pipeline
        self.data_issues = self._detect_data_issues()

    def _detect_data_issues(self) -> Dict:
        """
        Identify and characterize data quality problems.
        
        Returns:
            Dictionary with issue counts and gap locations.
        """
        issues = {
            'missing_dates': 0,
            'zero_cases': 0,
            'consecutive_zeros': 0,
            'data_gaps': []
        }

        if 'total_cases' in self.real_data.columns:
            # Identify weeks with zero reported cases
            # some of these are probably missing data, not actually zero tbh
            issues['zero_cases'] = (self.real_data['total_cases'] == 0).sum()

            # Count consecutive zeros
            if len(self.real_data) > 1:
                zero_mask = self.real_data['total_cases'] == 0
                consecutive_zeros = 0
                max_consecutive = 0

                for i in range(len(zero_mask)):
                    if zero_mask.iloc[i]:
                        consecutive_zeros += 1
                        max_consecutive = max(max_consecutive, consecutive_zeros)
                    else:
                        consecutive_zeros = 0

                issues['consecutive_zeros'] = max_consecutive

        if 'date' in self.real_data.columns and len(self.real_data) > 1:
            # Identify temporal gaps exceeding one week
            date_diff = self.real_data['date'].diff()
            gaps = date_diff[date_diff > pd.Timedelta(days=8)]

            for idx in gaps.index:
                gap_days = date_diff.loc[idx].days
                issues['data_gaps'].append({
                    'position': idx,
                    'gap_days': gap_days,
                    'missing_weeks': gap_days // 7 - 1
                })

            issues['missing_dates'] = len(gaps)

        return issues

    def _learn_patterns(self) -> Dict:
        """
        Learn key epidemiological patterns from real data.
        
        Analyzes seasonality cycles, outbreak characteristics, and climate relationships
        to inform synthetic data generation.
        
        Returns:
            Dictionary containing learned seasonal, outbreak, and climate patterns.
        """
        patterns = {}

        # 1. Seasonal Patterns (Weekly means)
        patterns['weekly_pattern'] = self.real_data.groupby('week_of_year')['total_cases'].mean().to_dict()

        # 2. Overall Case Statistics
        patterns['case_stats'] = {
            'mean': self.real_data['total_cases'].mean(),
            'std': self.real_data['total_cases'].std(),
            'median': self.real_data['total_cases'].median(),
            'max': self.real_data['total_cases'].max()
        }

        # 3. Outbreak Characteristics (thresholding at 90th percentile)
        outbreak_threshold = self.real_data['total_cases'].quantile(0.90)
        outbreak_mask = self.real_data['total_cases'] > outbreak_threshold

        if outbreak_mask.any():
            outbreak_series = self.real_data.loc[outbreak_mask, 'total_cases']
            patterns['outbreak_stats'] = {
                'threshold': outbreak_threshold,
                'frequency': outbreak_mask.mean(),
                'mean_intensity': outbreak_series.mean() - outbreak_threshold,
                'std_intensity': outbreak_series.std()
            }
        else:
            patterns['outbreak_stats'] = None

        # 4. Climate relationships
        patterns['climate_relationships'] = {}
        for climate_var in ['temperature', 'precipitation', 'humidity']:
            if climate_var in self.real_data.columns:
                patterns['climate_relationships'][climate_var] = {
                    'mean': self.real_data[climate_var].mean(),
                    'std': self.real_data[climate_var].std()
                }

        # 5. Seasonal Climate Patterns
        patterns['climate_weekly'] = {}
        for climate_var in ['temperature', 'precipitation', 'humidity', 'wind_speed']:
            if climate_var in self.real_data.columns:
                patterns['climate_weekly'][climate_var] = (
                    self.real_data.groupby('week_of_year')[climate_var].mean().to_dict()
                )

        return patterns

    def fix_data_issues(
        self,
        method: str = "interpolate",
        max_gap_weeks: int = 4
    ) -> pd.DataFrame:
        """
        Fix holes, zeros, and gaps in the existing real data.
        
        Note: This is experimental and changes the 'real' values.
        
        Args:
            method: Strategy for imputation ('interpolate', 'forward_fill', 'seasonal', 'hybrid')
            max_gap_weeks: Maximum consecutive gap to repair
        
        Returns:
            Processed dataframe with filled gaps.
        """
        fixed_data = self.real_data.copy()

        print(f"\n{'='*50}")
        print(f"!!! EXPERIMENTAL: FIXING DATA ISSUES !!!")
        print(f"{'='*50}")
        print(f"Zeros in cases: {self.data_issues['zero_cases']}")

        # Fix case data
        if 'total_cases' in fixed_data.columns:
            original_cases = fixed_data['total_cases'].copy()

            if method == "interpolate":
                fixed_data['total_cases'] = self._interpolate_cases(
                    original_cases, max_gap_weeks
                )
            elif method == "forward_fill":
                fixed_data['total_cases'] = original_cases.replace(0, np.nan).ffill(
                    limit=max_gap_weeks
                )
            elif method == "seasonal":
                fixed_data['total_cases'] = self._seasonal_fill(
                    original_cases, fixed_data
                )
            elif method == "hybrid":
                fixed_data['total_cases'] = self._hybrid_fill(
                    original_cases, fixed_data, max_gap_weeks
                )

            fixed_mask = (original_cases == 0) & (fixed_data['total_cases'] != 0)
            print(f"Fixed {fixed_mask.sum()} zero case entries")

        # Fix date gaps (add missing weeks)
        if self.data_issues['data_gaps']:
            fixed_data = self._fix_date_gaps(fixed_data, method, max_gap_weeks)

        # Fix climate variables
        for climate_var in ['temperature', 'precipitation', 'humidity']:
            if climate_var in fixed_data.columns:
                fixed_data[climate_var] = fixed_data[climate_var].interpolate(
                    limit=max_gap_weeks
                )

        return fixed_data

    def _interpolate_cases(self, cases: pd.Series, max_gap: int) -> pd.Series:
        """
        Interpolate zeros and small gaps in case data using linear interpolation.
        
        Args:
            cases: Original case series with potential gaps
            max_gap: Maximum consecutive missing values to interpolate
        
        Returns:
            Interpolated series with filled gaps.
        """
        cases_interp = cases.replace(0, np.nan)
        cases_interp = cases_interp.interpolate(
            method='linear',
            limit=max_gap,
            limit_direction='both'
        )
        cases_interp = cases_interp.ffill().bfill()
        return np.maximum(np.round(cases_interp), 0).astype(int)

    def _seasonal_fill(self, cases: pd.Series, df: pd.DataFrame) -> pd.Series:
        """
        Fill missing data using seasonal (week-of-year) averages from real data.
        
        Args:
            cases: Original case series with gaps
            df: Full dataframe with week_of_year column for context
        
        Returns:
            Series with seasonal gaps filled.
        """
        cases_filled = cases.copy()

        if 'week_of_year' in df.columns:
            weekly_avg = df[cases != 0].groupby('week_of_year')['total_cases'].mean()

            for idx in df[cases == 0].index:
                week = df.loc[idx, 'week_of_year']
                if week in weekly_avg.index:
                    cases_filled.iloc[idx] = weekly_avg[week]
                else:
                    cases_filled.iloc[idx] = df[cases != 0]['total_cases'].median()

        return np.maximum(np.round(cases_filled), 0).astype(int)

    def _hybrid_fill(self, cases: pd.Series, df: pd.DataFrame, max_gap: int) -> pd.Series:
        """
        Adaptive gap filling: linear interpolation for small gaps, seasonal for large gaps.
        
        Args:
            cases: Original case series with gaps
            df: Full dataframe context with temporal information
            max_gap: Threshold for small vs large gaps (weeks)
        
        Returns:
            Series with gaps filled using appropriate strategy.
        """
        cases_filled = cases.copy()

        # Find consecutive zero streaks
        zero_streaks = []
        in_streak = False
        streak_start = 0

        for i in range(len(cases)):
            if cases.iloc[i] == 0 and not in_streak:
                streak_start = i
                in_streak = True
            elif cases.iloc[i] != 0 and in_streak:
                streak_length = i - streak_start
                if streak_length > 0:
                    zero_streaks.append((streak_start, i - 1, streak_length))
                in_streak = False

        # Handle end of series
        if in_streak:
            streak_length = len(cases) - streak_start
            zero_streaks.append((streak_start, len(cases) - 1, streak_length))

        # Apply strategies based on gap size
        for start, end, length in zero_streaks:
            if length <= max_gap:
                # Small gap: interpolate between neighbors
                if start > 0 and end < len(cases) - 1:
                    prev_val = cases.iloc[start - 1]
                    next_val = cases.iloc[end + 1] if end < len(cases) - 1 else prev_val

                    for i in range(start, end + 1):
                        progress = (i - start + 1) / (length + 1)
                        cases_filled.iloc[i] = prev_val + (next_val - prev_val) * progress
            else:
                # Large gap: use seasonal pattern (interpolation too risky here)
                if 'week_of_year' in df.columns:
                    weekly_avg = df[cases != 0].groupby('week_of_year')['total_cases'].mean()
                    for i in range(start, end + 1):
                        week = df.loc[i, 'week_of_year']
                        cases_filled.iloc[i] = weekly_avg.get(
                            week, df[cases != 0]['total_cases'].median()
                        )

        return np.maximum(np.round(cases_filled), 0).astype(int)

    def _fix_date_gaps(self, df: pd.DataFrame, method: str, max_gap_weeks: int) -> pd.DataFrame:
        """
        Add missing dates to fill gaps in the time series.
        
        Args:
            df: Input dataframe with possible temporal gaps
            method: Imputation strategy for new rows ('interpolate', 'forward_fill', 'seasonal')
            max_gap_weeks: Maximum gap threshold for interpolation
        
        Returns:
            Dataframe with complete temporal coverage.
        """
        if len(df) < 2 or 'date' not in df.columns:
            return df

        # Create complete weekly date range
        full_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='W')

        if len(full_dates) == len(df):
            return df

        print(f"Filling {len(full_dates) - len(df)} missing weeks (ugh)")
        # not glamorous, but complete date grids save us later

        # pandas is so op
        df = (
            df.set_index('date')
            .reindex(full_dates)
            .reset_index()
            .rename(columns={'index': 'date'})
        )

        for col in df.columns:
            if col not in ['date', 'is_synthetic']:
                if method == "interpolate":
                    df[col] = df[col].interpolate(method='linear', limit=max_gap_weeks)

                df[col] = df[col].ffill(limit=max_gap_weeks).bfill(limit=max_gap_weeks)

        return df

    def create_farrington_ready_dataset(
        self,
        target_years: int = 7,
        region: str = "karnataka"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Extend dengue data backward to meet Farrington algorithm requirements.
        
        Args:
            target_years: Target historical span in years (minimum 5 recommended)
            region: Geographic region for monsoon pattern alignment
        
        Returns:
            Tuple of (extended dataframe, metadata dictionary with row counts).
        """
        current_years = len(self.real_data) / 52
        years_to_add = max(0, int(target_years - current_years))

        if years_to_add <= 0:
            return self.real_data, {"extension": "none"}

        logger.info(f"Adding {years_to_add} years of synthetic history...")

        # Project backwards from the earliest real data point
        new_data_list = []
        first_date = self.real_data['date'].min()

        # Generate synthetic weeks: Index 0 = Most Recent, Index N = Oldest
        for i in range(1, years_to_add * 52 + 1):
            new_date = first_date - pd.Timedelta(weeks=i)
            week_of_year = new_date.isocalendar().week

            new_row = {
                'date': new_date,
                'year': new_date.year,
                'week_of_year': int(week_of_year),
                'month': new_date.month,
                'is_synthetic': True
            }

            # 1. Base Seasonal Weather (Background)
            for var, rel in self.patterns.get('climate_relationships', {}).items():
                weekly_stats = self.patterns.get('climate_weekly', {}).get(var, {})
                seasonal_mean = weekly_stats.get(week_of_year, rel['mean'])
                # Add background noise
                new_row[var] = np.random.normal(seasonal_mean, rel['std'] * 0.5)

            # 2. Base Seasonal Cases
            base_cases = self.patterns['weekly_pattern'].get(
                week_of_year, self.patterns['case_stats']['median']
            )

            # Dampen variability in Dry Season (Jan-May)
            season_factor = 1.0
            if week_of_year < 20:
                season_factor = 0.7

            # Add background noise (scaled by season)
            noise = np.random.normal(0, self.patterns['case_stats']['std'] * 0.2 * season_factor)
            new_row['total_cases'] = max(0, int(base_cases * season_factor + noise))

            new_data_list.append(new_row)

        # 3. Inject Outbreaks AND Causal Weather Precursors
        for i in range(len(new_data_list)):
            row = new_data_list[i]
            week_of_year = row['week_of_year']

            # Decision: Is this an outbreak?
            # Only allow major outbreaks during Monsoon/Post-Monsoon (Weeks 20-52)
            is_dengue_season = (20 <= week_of_year <= 52)

            is_outbreak = False
            if is_dengue_season and self.patterns['outbreak_stats'] and np.random.random() < self.patterns['outbreak_stats']['frequency']:
                is_outbreak = True

            if is_outbreak:
                # Add Case Spike
                outbreak_boost = np.random.normal(
                    self.patterns['outbreak_stats']['mean_intensity'],
                    self.patterns['outbreak_stats']['std_intensity'] * 0.5
                )
                new_data_list[i]['total_cases'] += int(max(0, outbreak_boost))

                # INJECT PRECURSOR: Stochastically spike rain 4 weeks earlier
                prec_idx = i + 4
                if prec_idx < len(new_data_list) and np.random.random() < 0.65:
                    target_row = new_data_list[prec_idx]

                    # Boost Precipitation (Driver of mosquitoes)
                    if 'precipitation' in target_row:
                        rel = self.patterns['climate_relationships']['precipitation']
                        rain_boost = rel['std'] * np.random.uniform(1.0, 2.0)
                        target_row['precipitation'] += rain_boost

                    # Boost Humidity (Supportive condition)
                    if 'humidity' in target_row:
                        rel = self.patterns['climate_relationships']['humidity']
                        hum_boost = rel['std'] * np.random.uniform(0.5, 1.0)
                        target_row['humidity'] += hum_boost

        # Create synthetic history and sort it
        synthetic_df = pd.DataFrame(new_data_list).sort_values('date')

        combined_df = pd.concat([synthetic_df, self.real_data], ignore_index=True)
        combined_df = combined_df.sort_values('date').reset_index(drop=True)

        metadata = {
            "years_added": years_to_add,
            "total_rows": len(combined_df),
            "extension_type": "past_prepend"
        }

        return combined_df, metadata

    def create_complete_dataset(
        self,
        target_years: int = 7,
        fix_method: str = "hybrid",
        region: str = "karnataka",
        fix_existing: bool = False,
        extend_history: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete data transformation pipeline: repair gaps and extend history.
        
        Args:
            target_years: Target historical span for Farrington (years)
            fix_method: Data repair strategy ('interpolate', 'seasonal', 'hybrid')
            region: Geographic region for monsoon alignment
            fix_existing: Whether to repair existing data gaps
            extend_history: Whether to synthesize backward history
        
        Returns:
            Tuple of (complete dataframe ready for modeling, detailed metadata dictionary).
        """
        working_data = self.real_data.copy()
        metadata = {'steps': [], 'fixes_made': {}}

        # Step 1: Fix existing data issues
        if fix_existing:
            print(f"\nStep 1: Fixing holes...")
            if self.data_issues['zero_cases'] > 0 or self.data_issues['missing_dates'] > 0:
                working_data = self.fix_data_issues(method=fix_method)
        else:
            print("\nStep 1: Skipping hole fixing (playing it safe)")

        # Step 2: Extend history
        if extend_history:
            logger.info(f"Step 2: Extending history to {target_years} years")
            extender = DengueDataExtender(working_data, seed=self.seed)
            extended_data, ext_metadata = extender.create_farrington_ready_dataset(target_years=target_years, region=region)

            # Mark data types correctly (synthetic history comes before real data)
            real_len = len(working_data)
            extended_data['data_type'] = 'synthetic'
            col_idx = extended_data.columns.get_loc('data_type')
            extended_data.iloc[-real_len:, col_idx] = 'real'

            working_data = extended_data

            metadata['synthetic_rows'] = len(working_data) - real_len
            metadata['real_data_rows'] = real_len
            metadata.update(ext_metadata)
        else:
            working_data['data_type'] = 'real'
            metadata['synthetic_rows'] = 0
            metadata['real_data_rows'] = len(working_data)

        # Step 3: Create final spike labels
        mean_52 = working_data['total_cases'].rolling(window=52, min_periods=1).mean()
        std_52 = working_data['total_cases'].rolling(window=52, min_periods=1).std()
        working_data['spike_2sigma'] = ((working_data['total_cases'] - mean_52) > 2 * std_52).astype(int)

        print(f"\n{'='*50}")
        print(f"DONE! Dataset Ready.")
        print(f"{'='*50}")
        print(f"Total rows: {len(working_data)}")
        print(f"Spike rate: {working_data['spike_2sigma'].mean():.2%}")
        # oh boy this took awhile

        return working_data, metadata