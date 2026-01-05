import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class ModelConfig:
    """
    Configuration for dengue outbreak prediction model.
    Optimized for epidemiological time-series.
    Deep trees to capture complex temporal patterns.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.default_params = {
            'n_estimators': 500,          # Number of trees in the forest
            'max_depth': None,              # Maximum depth of each tree
            'min_samples_split': 2,       # Minimum samples to split a node
            'min_samples_leaf': 1,        # Minimum samples at a leaf node
            'max_features': 'sqrt',       # Features to consider at each split
            'class_weight': 'balanced',   # Handle class imbalance
            'random_state': 42,          # For reproducibility
            'n_jobs': -1,            # Use all available cores
            'bootstrap': True,              # Bootstrap samples for trees
            'oob_score': False,             # Use out of bag samples to estimate accuracy
            'max_samples': 0.8,          # Use 80% of data for training each tree
            'min_impurity_decrease': 0.0,   # Minimum impurity decrease for splits
            'ccp_alpha': 0.0
        }

        if params:
            self.default_params.update(params)
    def to_dict(self) -> Dict:
        return self.default_params.copy()

# ============================================================================
# MODEL CLASS
# ============================================================================

class DengueRandomForest:
    """
    Random Forest classifier for dengue outbreak prediction
    
    Features:
    - Time-series cross-validation (no data leakage)
    - Hyperparameter tuning optimized for outbreak detection
    - Probability calibration for reliable risk estimates
    - Permutation importance for feature interpretation
    - Metadata tracking for reproducibility
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.training_date: Optional[str] = None
        self.cv_scores: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {
            'pipeline_version': '1.0',
            'run_timestamp': datetime.now().isoformat(),
            'config': config.to_dict()
        }
    def train(self, x: pd.DataFrame, y: pd.Series, tune: bool = True, n_iter: int = 30, calibrate: bool = True) -> None:
        """
        Train the Random Forest model with time-series cross-validation.
        
        Args:
            x: Feature dataframe (Needs to be chronologically ordered)
            y: Target series (1 = outbreak next week, 0 = no outbreak)
            tune: Whether to perform hyperparameter tuning
            n_iter: Number of iterations for random search
            calibrate: Whether to calibrate probability estimates
        """
        logger.info("Starting model training")
        logger.info(f"Data Shape: {x.shape}, Outbeak Prevalence: {y.mean():.2%}")

        self._validate_inputs(x, y)
        self.feature_names = x.columns.tolist()

        # Use TimeSeriesSplit for training to prevent data leakage
        tscv = TimeSeriesSplit(n_splits=5, test_size=52, gap=0)  # Test on 1 year of data

        if tune:
            self.model = self._tune_hyperparams(x, y, tscv, n_iter)
        else:
            logger.info("Using default model parameters")
            self.model = RandomForestClassifier(**self.config.default_params)

            # Manual cross-validation for training
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(x)):
                x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                self.model.fit(x_train, y_train)
                y_pred = self.model.predict(x_val)
                score = f1_score(y_val, y_pred, zero_division=0)
                cv_scores.append(score)
                logger.info(f"Fold {fold+1} F1 Score: {score:.3f}")
            self.cv_scores = np.array(cv_scores)
            self.metadata['cv_mode'] = 'manual'
            logger.info(f"CV F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

            # Final Fit on all the data
            self.model.fit(x, y)

        if calibrate and self.model is not None:
            logger.info("Calibrating model estimates with TimeSeriesSplit")
            self.model = CalibratedClassifierCV(
                self.model,
                method="isotonic",
                cv=TimeSeriesSplit(n_splits=3, test_size=26, gap=0)  # Test on 6 months for calibration
            )
            self.model.fit(x, y)
        self.training_date = datetime.now().strftime("%Y-%m-%d")
        self.metadata['cv_mean_f1'] = float(np.mean(self.cv_scores)) if self.cv_scores is not None else None
        self.metadata['calibrated'] = calibrate

        logger.info("Model training complete!")
        self._print_feature_importance()
    
    def _tune_hyperparams(self, x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit, n_iter: int) -> RandomForestClassifier:
        """
        Hyperparameter tuning with randomized search.
        Optimized for outbreak detection in time series data.
        """
        logger.info(f"Starting hyperparameter tuning ({n_iter} iterations)")

        param_dist = {
            'n_estimators': [300, 500, 800, 1000],
            'max_depth': [None, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True],
            'max_samples': [0.6, 0.8, 1.0]
        }

        rf = RandomForestClassifier(
            class_weight= self.config.default_params['class_weight'],
            random_state=42,
            n_jobs=-1
        )

        search = RandomizedSearchCV(
            rf,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        search.fit(x,y)

        self.cv_scores = search.cv_results_['mean_test_score']
        self.metadata['cv_mode'] = 'tuning'

        logger.info(f"Best F1: {search.best_score_:.3f} with params: {search.best_params_}")

        return search.best_estimator_
    
    def _validate_inputs(self, x: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate input data for  modeling.
        """
        if len(x) != len(y):
            raise ValueError(f"x and y length mismatch: {len(x)} vs {len(y)}")
        
        if x.isnull().any().any():
            null_cols = x.columns[x.isnull().any()].tolist()
            raise ValueError(f"NaN values in x columns: {null_cols}")
        
        if y.isnull().any():
            raise ValueError(f"NaN values in y: {y.isnull().sum()} rows")
        
        if not isinstance(x.index, pd.DatetimeIndex):
            logger.warning("x index is not DatetimeIndex; Make sure data is chronologically ordered to prevent data leakage.")

        # Check outbreak prevalence
        outbreak_rate = y.mean()
        if outbreak_rate < 0.05 or outbreak_rate > 0.5:
            logger.warning(f"Unusual outbreak prevalence: {outbreak_rate:.2%}. Check data quality.")
    
    def _print_feature_importance(self, top_n: int = 15) -> None:
        """
        Print Gini feature importance from trained model.
        """
        if self.model is None:
            return
        
        # Handle calibrated model wrapper
        if hasattr(self.model, 'calibrated_classifiers_'):
            base_model = self.model.calibrated_classifiers_[0].estimator
        else:
            base_model = self.model

        importances = base_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        logger.info(f"\n{'='*50}")
        logger.info("GINI FEATURE IMPORTANCE")
        logger.info(f"{'='*50}")

        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            logger.info(f"{i+1:2d}. {self.feature_names[idx]:25s} {importances[idx]:.4f}")

    def compute_permutation_importance(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_repeats: int = 10
    ) -> Dict[str, np.ndarray]:

        if self.model is None:
            raise ValueError("Model is not trained")

        estimator = (
            self.model.base_estimator
            if hasattr(self.model, "base_estimator")
            else self.model
        )

        result = permutation_importance(
            estimator,
            x_test,
            y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
            scoring='f1'
        )

        indices = np.argsort(result.importances_mean)[::-1]

        for i in range(min(15, len(indices))):
            idx = indices[i]
            logger.info(
                f"{i+1:2d}. {self.feature_names[idx]:25s} "
                f"{result.importances_mean[idx]:.4f} ± {result.importances_std[idx]:.4f}"
            )

        return {
            'importances_mean': result.importances_mean,
            'importances_std': result.importances_std,
            'indices': indices
        }

    def predict(self, x: pd.DataFrame, proba: bool = False, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions with optional probability output
        
        Args:
            x: Feature matrix
            proba: If True, return probabilities instead of binary predictions
            threshold: Classification threshold for binary predictions
        
        Returns:
            Predictions (binary or probabilities)
        """
        if self.model is None:
            raise ValueError("Model is not trained")
        
        if proba:
            probs = self.model.predict_proba(x)[:, 1]
            return probs
        else:
            probs = self.model.predict_proba(x)[:, 1]
            return (probs >= threshold).astype(int)
        
    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Returns:
            Dictionary with F1, precision, recall, AUROC
        """
        y_pred = self.predict(x_test, threshold=threshold)
        try:
            y_proba = self.model.predict_proba(x_test)[:, 1]
        except AttributeError:
            # Fallback to base estimator
            y_proba = self.model.base_estimator.predict_proba(x_test)[:, 1]

        metrics = {
            'outbreak_rate': y_test.mean(),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'pred_outbreak_rate': y_pred.mean(),
            'auroc': roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else np.nan
        }


        logger.info(f"\n{'='*50}")
        logger.info("TEST SET PERFORMANCE")
        logger.info(f"{'='*50}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Actual outbreak rate: {metrics['outbreak_rate']:.2%}")
        logger.info(f"Predicted outbreak rate: {metrics['pred_outbreak_rate']:.2%}")

        for k, v in metrics.items():
            if k not in ['outbreak_rate', 'pred_outbreak_rate']:
                logger.info(f"{k:>12s}: {v:.3f}")

        return metrics
    
    def save(self, path: str = "models/rf_dengue.pkl") -> None:
        """
        Save model with full metadata for reproducibility.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            'model': self.model,
            'config': self.config.to_dict(),
            'feature_names': self.feature_names,
            'training_date': self.training_date,
            'cv_scores': self.cv_scores.tolist() if self.cv_scores is not None else None,
            'metadata': self.metadata
        }

        joblib.dump(data, path)
        logger.info(f"Model saved to {path}")
        logger.info(f"Features: {len(self.feature_names)}, CV F1: {self.metadata.get('cv_mean_f1', 'N/A')}")

    @classmethod
    def load(cls, path: str = "models/rf_dengue.pkl") -> 'DengueRandomForest':
        """
        Load saved model with validation.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)

        # Validate loaded data
        required_keys = ['model', 'config', 'feature_names', 'training_date', 'metadata']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"the key '{key}' is missing in the loaded model data")
        
        config_params = data.get('config', {})
        instance = cls(ModelConfig(config_params))
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.training_date = data['training_date']
        instance.cv_scores = np.array(data['cv_scores']) if data.get('cv_scores') is not None else None
        instance.metadata = data['metadata']

        logger.info(f"Model loaded from {path}")
        logger.info(f"Training Date: {instance.training_date}")
        logger.info(f"Features: {len(instance.feature_names)}")

        return instance
    
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_model_with_custom_weights(x: pd.DataFrame, y: pd.Series, outbreak_weight: float = 3.0) -> DengueRandomForest:
    """
    Create Random Forest model with custom class weights
    
    Args:
        x: Features
        y: Targets
        outbreak_weight: Weight for outbreak class (1 = no outbreak)
    
    Returns:
        Trained RandomForestClassifier
    """

    # Calculate class weights
    n_samples = len(y)
    n_outbreaks = y.sum()
    n_non_outbreaks = n_samples - n_outbreaks

    weight_0 = n_samples / (2 * n_non_outbreaks) # Non-outbreak weight
    weight_1 = outbreak_weight * weight_0        # Outbreak weight

    logger.info(f"Class weights: Non-outbreak={weight_0:.2f}, Outbreak={weight_1:.2f}")

    config = ModelConfig({
        'class_weight': {0: weight_0, 1: weight_1}
    })

    model = DengueRandomForest(config)
    model.train(x, y, tune=True, calibrate=True)

    return model

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the modeling module.
    """

    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Features with temporal autocorrelation (close to real disease data)
    x = pd.DataFrame()
    for i in range(n_features):
        # Create AR(1) process for realistic time-series
        noise = np.random.randn(n_samples) * 0.5
        if i == 0:
            x[f'cases_momentum'] = 0.7 * np.random.randn(n_samples) + noise
        elif i < 4:
            x[f'temp_lag{i}'] = 0.8 * np.random.randn(n_samples) + noise
        else:
            x[f'precip_lag{i-3}'] = 0.6 * np.random.randn(n_samples) + noise
    
    # Add date index for time-series validation
    dates = pd.date_range(start='2010-01-01', periods=n_samples, freq='W')
    x.index = dates
    
    # Create targets with outbreak clusters (realistic)
    y = np.zeros(n_samples)
    outbreak_prob = 0.13  # 13% outbreak prevalence
    
    # Create clustered outbreaks (dengue is not random)
    for t in range(20, n_samples-1):
        if np.random.random() < outbreak_prob / 10:  # Start outbreak
            outbreak_length = np.random.randint(2, 6)
            y[t:t+outbreak_length] = 1
    
    y = pd.Series(y, index=dates)
    
    logger.info(f"Synthetic data: {x.shape}, Outbreak rate: {y.mean():.2%}")
    
    # Create and train model
    model = create_model_with_custom_weights(x, y, outbreak_weight=3.0)
    
    # Evaluate (in real usage, use proper train/test split)
    metrics = model.evaluate(x.iloc[-200:], y.iloc[-200:])
    
    # Compute permutation importance
    perm_results = model.compute_permutation_importance(x.iloc[-200:], y.iloc[-200:])
    
    # Save model
    model.save("example_model.pkl")
    
    # Load and verify
    loaded_model = DengueRandomForest.load("example_model.pkl")
    
    logger.info("Example completed successfully")
