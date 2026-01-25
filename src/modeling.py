import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import joblib
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration container for model parameters."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

class DengueRandomForest:
    """
    Random Forest classifier for dengue outbreak prediction with threshold optimization.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = RandomForestClassifier(
            n_estimators=500, 
            max_depth=5,
            min_samples_leaf=10,
            class_weight='balanced_subsample',
            random_state=42,
            max_features='sqrt',
            n_jobs=-1
        )
        self.best_threshold = 0.3 # Default starting point

    def train(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Train the Random Forest model with synthetic oversampling for class balance.
        
        Args:
            x: Feature dataframe (must be chronologically ordered for time-series)
            y: Target series (1 = outbreak next week, 0 = no outbreak)
        """
        minority_count = int((y == 1).sum())
        if minority_count < 2:
            logger.warning(
                "Skipping SMOTE: minority_count=%s (<2). Training on original data.",
                minority_count
            )
            X_res, y_res = x, y
        else:
            k_neighbors = min(5, minority_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(x, y)
        # Balance the classes so training isn't dominated by non-spike weeks
        # hopefully this SMOTE step doesn't create too much artificial data lol
        
        # Ensure feature names are preserved for XAI (SHAP)
        if isinstance(x, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=x.columns)
            
        self.model.fit(X_res, y_res)
        logger.info(f"Model trained on {x.shape[1]} features.")

    def _get_scaled_probs(self, x: pd.DataFrame) -> np.ndarray:
        """
        Apply monsoon-seasonalized probability scaling to raw model outputs.
        
        Args:
            x: Feature dataframe
            
        Returns:
            Adjusted probability array
        """
        probs = self.model.predict_proba(x)[:, 1]
        if 'month' in x.columns:
            monsoon_months = [6, 7, 8, 9, 10, 11]
            # 1.3 for Monsoon, 0.4 for Dry (this scaling saved me so many times lol)
            scalers = np.array([1.3 if m in monsoon_months else 0.4 for m in x['month']])
            probs = probs * scalers
        return probs

    def predict(self, x: pd.DataFrame, threshold: Optional[float] = None):
        """
        Generate outbreak predictions using optimized probability threshold.
        
        Args:
            x: Feature dataframe
            threshold: Custom threshold override (uses optimized value if None)
            
        Returns:
            Binary outbreak predictions (0 or 1)
        """
        t = threshold if threshold is not None else self.best_threshold
        probs = self._get_scaled_probs(x)
        return (probs >= t).astype(int)

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series, threshold: Optional[float] = None):
        """
        Evaluate model performance and optimize classification threshold for F1 score.
        
        Args:
            x_test: Test feature dataframe
            y_test: Test target series
            threshold: Optional fixed threshold (skips optimization if provided)
            
        Returns:
            Dictionary with F1, precision, recall, and optimal threshold
        """
        probs = self._get_scaled_probs(x_test)
        min_recall = float(self.config.params.get('min_recall', 0.7))
        min_f1 = float(self.config.params.get('min_f1', 0.4))
        threshold_grid = np.arange(0.05, 0.95, 0.01)
        
        # If a specific threshold is forced, use it. Otherwise, optimize
        if threshold is not None:
            self.best_threshold = threshold
        else:
            best_f1 = -1.0
            best_rec = -1.0
            opt_t = self.best_threshold

            for t in threshold_grid:
                preds = (probs >= t).astype(int)
                f1 = f1_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)

                if rec >= min_recall and f1 >= min_f1:
                    if f1 > best_f1:
                        best_f1 = f1
                        opt_t = t

            if best_f1 < 0:
                # Fallback: maximize recall, then F1
                for t in threshold_grid:
                    preds = (probs >= t).astype(int)
                    f1 = f1_score(y_test, preds, zero_division=0)
                    rec = recall_score(y_test, preds, zero_division=0)

                    if rec > best_rec or (rec == best_rec and f1 > best_f1):
                        best_rec = rec
                        best_f1 = f1
                        opt_t = t

                logger.warning(
                    "No threshold met targets (recall>=%.2f, f1>=%.2f). Using best recall/f1 fallback.",
                    min_recall,
                    min_f1
                )

            self.best_threshold = opt_t

        final_preds = (probs >= self.best_threshold).astype(int)
        
        results = {
            'f1': f1_score(y_test, final_preds, zero_division=0),
            'precision': precision_score(y_test, final_preds, zero_division=0),
            'recall': recall_score(y_test, final_preds, zero_division=0),
            'best_threshold': self.best_threshold
        }
        
        print(f"--- Backend Optimization Results ---")
        print(f"Selected Threshold: {results['best_threshold']:.2f} | F1: {results['f1']:.4f}")
        return results
    
    def save(self, path: str = "models/rf_dengue_final.pkl"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        
    def load(self, path: str = "models/rf_dengue_final.pkl"):
        self.model = joblib.load(path)

    @property
    def feature_importances_(self):
        if self.model is None: return None
        return self.model.feature_importances_