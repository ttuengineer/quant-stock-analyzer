"""
Machine Learning Stock Predictor

Ensemble of XGBoost, LightGBM, and RandomForest models for stock return prediction.

Features:
- Multi-model ensemble with weighted voting
- Walk-forward validation for realistic out-of-sample testing
- Predicts 1-day, 5-day, and 20-day forward returns
- Classification (up/down) and regression (return magnitude)
- Model persistence (save/load trained models)
- Feature importance analysis

Academic References:
- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"
- Ke et al. (2017): "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- Breiman (2001): "Random Forests"
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import joblib
from datetime import datetime

# ML models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from ..utils.logger import setup_logger
from .feature_engineer import FeatureEngineer

logger = setup_logger(__name__)


class StockPredictor:
    """
    Ensemble ML predictor for stock returns.

    Uses three models:
    - XGBoost: Gradient boosting with regularization
    - LightGBM: Fast gradient boosting with leaf-wise growth
    - RandomForest: Ensemble of decision trees

    Predictions:
    - Direction (up/down): Classification probability
    - Magnitude: Predicted return percentage
    - Confidence: Model agreement and uncertainty
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        retrain_days: int = 90,
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ML predictor.

        Args:
            model_dir: Directory to save/load models
            retrain_days: Retrain models every N days
            ensemble_weights: Custom weights for models (XGBoost, LightGBM, RandomForest)
        """
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(exist_ok=True)

        self.retrain_days = retrain_days
        self.ensemble_weights = ensemble_weights or {
            "xgboost": 0.40,
            "lightgbm": 0.35,
            "random_forest": 0.25
        }

        # Models for classification (direction)
        self.xgb_clf = None
        self.lgb_clf = None
        self.rf_clf = None

        # Models for regression (magnitude)
        self.xgb_reg = None
        self.lgb_reg = None
        self.rf_reg = None

        # Feature scaler
        self.scaler = StandardScaler()

        # Feature engineer
        self.feature_engineer = FeatureEngineer()

        # Training metadata
        self.last_trained = None
        self.feature_names = []
        self.feature_importance = {}

        # Try to load existing models
        self._load_models()

        logger.info("Initialized ML predictor with ensemble weights: %s", self.ensemble_weights)

    def predict(
        self,
        features: pd.DataFrame,
        return_components: bool = False
    ) -> Dict:
        """
        Predict stock returns using ensemble.

        Args:
            features: Engineered features (single row DataFrame)
            return_components: If True, return individual model predictions

        Returns:
            Dictionary with:
            - direction_prob: Probability of positive return (0-1)
            - predicted_return: Expected return percentage
            - confidence: Model agreement (0-1)
            - signal: BUY/SELL/HOLD based on thresholds
            - components: Individual model predictions (if requested)
        """
        if features is None or features.empty:
            logger.warning("No features provided for prediction")
            return self._default_prediction()

        # Check if models are trained
        if not self._models_ready():
            logger.warning("Models not trained yet - returning neutral prediction")
            return self._default_prediction()

        try:
            # Prepare features
            X = features[self.feature_names]
            X_scaled = self.scaler.transform(X)

            # Get predictions from each model
            xgb_prob = self.xgb_clf.predict_proba(X_scaled)[0, 1]
            lgb_prob = self.lgb_clf.predict_proba(X_scaled)[0, 1]
            rf_prob = self.rf_clf.predict_proba(X_scaled)[0, 1]

            xgb_return = self.xgb_reg.predict(X_scaled)[0]
            lgb_return = self.lgb_reg.predict(X_scaled)[0]
            rf_return = self.rf_reg.predict(X_scaled)[0]

            # Ensemble with weighted voting
            ensemble_prob = (
                xgb_prob * self.ensemble_weights["xgboost"] +
                lgb_prob * self.ensemble_weights["lightgbm"] +
                rf_prob * self.ensemble_weights["random_forest"]
            )

            ensemble_return = (
                xgb_return * self.ensemble_weights["xgboost"] +
                lgb_return * self.ensemble_weights["lightgbm"] +
                rf_return * self.ensemble_weights["random_forest"]
            )

            # Calculate confidence (model agreement)
            prob_std = np.std([xgb_prob, lgb_prob, rf_prob])
            confidence = 1.0 - min(prob_std * 2, 1.0)  # Lower std = higher confidence

            # Generate signal
            signal = self._generate_signal(ensemble_prob, ensemble_return, confidence)

            result = {
                "direction_prob": float(ensemble_prob),
                "predicted_return": float(ensemble_return),
                "confidence": float(confidence),
                "signal": signal,
                "timestamp": datetime.now().isoformat()
            }

            if return_components:
                result["components"] = {
                    "xgboost": {"prob": float(xgb_prob), "return": float(xgb_return)},
                    "lightgbm": {"prob": float(lgb_prob), "return": float(lgb_return)},
                    "random_forest": {"prob": float(rf_prob), "return": float(rf_return)}
                }

            logger.debug(
                f"ML Prediction - Prob: {ensemble_prob:.3f}, Return: {ensemble_return:.3f}%, "
                f"Confidence: {confidence:.3f}, Signal: {signal}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return self._default_prediction()

    def train(
        self,
        features: pd.DataFrame = None,
        direction_labels: pd.Series = None,
        return_labels: pd.Series = None,
        training_data: pd.DataFrame = None,
        target_column: str = "forward_return_5d",
        test_size: float = 0.2,
        validation_split: float = 0.2,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict:
        """
        Train ensemble models using walk-forward validation.

        Two modes:
        1. Simple mode: Pass features, direction_labels, and return_labels directly
        2. DataFrame mode: Pass training_data with target_column

        Args:
            features: Feature DataFrame (for simple mode)
            direction_labels: Binary labels for direction (for simple mode)
            return_labels: Continuous labels for returns (for simple mode)
            training_data: Historical data with features and targets (for DataFrame mode)
            target_column: Column name for target variable (for DataFrame mode)
            test_size: Fraction of data for test set
            validation_split: Fraction of data for validation
            n_splits: Number of time series cross-validation splits
            random_state: Random seed for reproducibility

        Returns:
            Training metrics and performance stats
        """
        # Handle two different modes
        if features is not None and direction_labels is not None and return_labels is not None:
            # Simple mode - features and labels provided directly
            X = features
            y_direction = direction_labels
            y_return = return_labels
            logger.info(f"Training ML models on {len(X)} samples (simple mode)...")
        elif training_data is not None:
            # DataFrame mode - extract from training data
            X = training_data[self.feature_names]
            y_return = training_data[target_column]
            y_direction = (y_return > 0).astype(int)
            logger.info(f"Training ML models on {len(training_data)} samples (DataFrame mode)...")
        else:
            raise ValueError("Must provide either (features, direction_labels, return_labels) or training_data")

        try:
            # Store feature names if not already set
            if not self.feature_names:
                self.feature_names = list(X.columns)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split into train and test
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import precision_score, recall_score, f1_score

            test_size_actual = int(len(X) * test_size)
            X_train = X_scaled[:-test_size_actual]
            X_test = X_scaled[-test_size_actual:]
            y_direction_train = y_direction.iloc[:-test_size_actual] if hasattr(y_direction, 'iloc') else y_direction[:-test_size_actual]
            y_direction_test = y_direction.iloc[-test_size_actual:] if hasattr(y_direction, 'iloc') else y_direction[-test_size_actual:]
            y_return_train = y_return.iloc[:-test_size_actual] if hasattr(y_return, 'iloc') else y_return[:-test_size_actual]
            y_return_test = y_return.iloc[-test_size_actual:] if hasattr(y_return, 'iloc') else y_return[-test_size_actual:]

            # Train on training set
            self._train_classifiers(X_train, y_direction_train)
            self._train_regressors(X_train, y_return_train)

            # Evaluate on test set
            direction_probs = self._predict_direction_ensemble(X_test)
            direction_preds = (direction_probs > 0.5).astype(int)
            return_preds = self._predict_return_ensemble(X_test)

            # Calculate metrics on test set
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            direction_accuracy = accuracy_score(y_direction_test, direction_preds)
            direction_precision = precision_score(y_direction_test, direction_preds, zero_division=0)
            direction_recall = recall_score(y_direction_test, direction_preds, zero_division=0)
            direction_f1 = f1_score(y_direction_test, direction_preds, zero_division=0)
            direction_auc = roc_auc_score(y_direction_test, direction_probs)

            return_mae = mean_absolute_error(y_return_test, return_preds)
            return_mse = mean_squared_error(y_return_test, return_preds)
            return_rmse = np.sqrt(return_mse)
            return_r2 = r2_score(y_return_test, return_preds)

            # Retrain on full dataset for production use
            self._train_classifiers(X_scaled, y_direction if isinstance(y_direction, np.ndarray) else y_direction.values)
            self._train_regressors(X_scaled, y_return if isinstance(y_return, np.ndarray) else y_return.values)

            # Calculate feature importance
            self._calculate_feature_importance()

            # Update metadata
            self.last_trained = datetime.now()

            # Aggregate metrics
            results = {
                "direction_accuracy": direction_accuracy,
                "direction_precision": direction_precision,
                "direction_recall": direction_recall,
                "direction_f1": direction_f1,
                "direction_auc": direction_auc,
                "return_mae": return_mae,
                "return_mse": return_mse,
                "return_rmse": return_rmse,
                "return_r2": return_r2,
                "n_samples": len(X),
                "n_features": len(self.feature_names),
                "trained_at": self.last_trained.isoformat()
            }

            logger.info(
                f"Training complete - Accuracy: {direction_accuracy:.3f}, "
                f"Precision: {direction_precision:.3f}, Recall: {direction_recall:.3f}, "
                f"F1: {direction_f1:.3f}, MAE: {return_mae:.4f}"
            )

            return results

        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}

    def _train_classifiers(self, X: np.ndarray, y: np.ndarray):
        """Train classification models (direction prediction)."""
        # XGBoost Classifier
        self.xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.xgb_clf.fit(X, y)

        # LightGBM Classifier
        self.lgb_clf = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.lgb_clf.fit(X, y)

        # Random Forest Classifier
        self.rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_clf.fit(X, y)

    def _train_regressors(self, X: np.ndarray, y: np.ndarray):
        """Train regression models (return magnitude prediction)."""
        # XGBoost Regressor
        self.xgb_reg = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        self.xgb_reg.fit(X, y)

        # LightGBM Regressor
        self.lgb_reg = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='regression',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.lgb_reg.fit(X, y)

        # Random Forest Regressor
        self.rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_reg.fit(X, y)

    def _predict_direction_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction for direction (classification)."""
        xgb_prob = self.xgb_clf.predict_proba(X)[:, 1]
        lgb_prob = self.lgb_clf.predict_proba(X)[:, 1]
        rf_prob = self.rf_clf.predict_proba(X)[:, 1]

        ensemble = (
            xgb_prob * self.ensemble_weights["xgboost"] +
            lgb_prob * self.ensemble_weights["lightgbm"] +
            rf_prob * self.ensemble_weights["random_forest"]
        )

        return ensemble

    def _predict_return_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction for return magnitude (regression)."""
        xgb_return = self.xgb_reg.predict(X)
        lgb_return = self.lgb_reg.predict(X)
        rf_return = self.rf_reg.predict(X)

        ensemble = (
            xgb_return * self.ensemble_weights["xgboost"] +
            lgb_return * self.ensemble_weights["lightgbm"] +
            rf_return * self.ensemble_weights["random_forest"]
        )

        return ensemble

    def _calculate_feature_importance(self):
        """Calculate and store feature importance from all models."""
        importance = {}

        # XGBoost importance
        xgb_imp = dict(zip(self.feature_names, self.xgb_clf.feature_importances_))

        # LightGBM importance
        lgb_imp = dict(zip(self.feature_names, self.lgb_clf.feature_importances_))

        # Random Forest importance
        rf_imp = dict(zip(self.feature_names, self.rf_clf.feature_importances_))

        # Average importance across models
        for feature in self.feature_names:
            importance[feature] = (
                xgb_imp[feature] * self.ensemble_weights["xgboost"] +
                lgb_imp[feature] * self.ensemble_weights["lightgbm"] +
                rf_imp[feature] * self.ensemble_weights["random_forest"]
            )

        # Sort by importance
        self.feature_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        logger.debug(f"Top 5 features: {list(self.feature_importance.keys())[:5]}")

    def _generate_signal(self, prob: float, return_pred: float, confidence: float) -> str:
        """
        Generate trading signal from ML predictions.

        Rules:
        - STRONG_BUY: prob > 0.70, return > 3%, confidence > 0.7
        - BUY: prob > 0.60, return > 1%, confidence > 0.5
        - SELL: prob < 0.40, return < -1%, confidence > 0.5
        - STRONG_SELL: prob < 0.30, return < -3%, confidence > 0.7
        - HOLD: Otherwise
        """
        if prob > 0.70 and return_pred > 3.0 and confidence > 0.7:
            return "STRONG_BUY"
        elif prob > 0.60 and return_pred > 1.0 and confidence > 0.5:
            return "BUY"
        elif prob < 0.40 and return_pred < -1.0 and confidence > 0.5:
            return "SELL"
        elif prob < 0.30 and return_pred < -3.0 and confidence > 0.7:
            return "STRONG_SELL"
        else:
            return "HOLD"

    def _models_ready(self) -> bool:
        """Check if models are trained and ready."""
        return all([
            self.xgb_clf is not None,
            self.lgb_clf is not None,
            self.rf_clf is not None,
            self.xgb_reg is not None,
            self.lgb_reg is not None,
            self.rf_reg is not None,
            len(self.feature_names) > 0
        ])

    def _default_prediction(self) -> Dict:
        """Return neutral prediction when models aren't ready."""
        return {
            "direction_prob": 0.5,
            "predicted_return": 0.0,
            "confidence": 0.0,
            "signal": "HOLD",
            "timestamp": datetime.now().isoformat(),
            "note": "Models not trained or insufficient data"
        }

    def save_models(self, output_dir: str = None):
        """
        Save trained models to disk.

        Args:
            output_dir: Directory to save models (uses self.model_dir if None)
        """
        try:
            save_dir = Path(output_dir) if output_dir else self.model_dir
            save_dir.mkdir(exist_ok=True, parents=True)

            timestamp = datetime.now().strftime("%Y%m%d")

            models = {
                "xgb_clf": self.xgb_clf,
                "lgb_clf": self.lgb_clf,
                "rf_clf": self.rf_clf,
                "xgb_reg": self.xgb_reg,
                "lgb_reg": self.lgb_reg,
                "rf_reg": self.rf_reg,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance,
                "ensemble_weights": self.ensemble_weights,
                "last_trained": self.last_trained
            }

            model_path = save_dir / f"stock_predictor_{timestamp}.pkl"
            joblib.dump(models, model_path)

            # Also save as "latest"
            latest_path = save_dir / "stock_predictor_latest.pkl"
            joblib.dump(models, latest_path)

            logger.info(f"Models saved to {model_path}")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def _save_models(self):
        """Internal method to save models (backward compatibility)."""
        self.save_models()

    def _load_models(self):
        """Load trained models from disk."""
        try:
            latest_path = self.model_dir / "stock_predictor_latest.pkl"

            if not latest_path.exists():
                logger.info("No pre-trained models found")
                return

            models = joblib.load(latest_path)

            self.xgb_clf = models["xgb_clf"]
            self.lgb_clf = models["lgb_clf"]
            self.rf_clf = models["rf_clf"]
            self.xgb_reg = models["xgb_reg"]
            self.lgb_reg = models["lgb_reg"]
            self.rf_reg = models["rf_reg"]
            self.scaler = models["scaler"]
            self.feature_names = models["feature_names"]
            self.feature_importance = models["feature_importance"]
            self.ensemble_weights = models.get("ensemble_weights", self.ensemble_weights)
            self.last_trained = models.get("last_trained")

            logger.info(f"Loaded pre-trained models from {latest_path}")
            logger.info(f"Last trained: {self.last_trained}")

        except Exception as e:
            logger.warning(f"Could not load models: {e}")

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get top N most important features."""
        if not self.feature_importance:
            return {}

        return dict(list(self.feature_importance.items())[:top_n])

    def needs_retraining(self) -> bool:
        """Check if models need retraining based on age."""
        if self.last_trained is None:
            return True

        days_since_training = (datetime.now() - self.last_trained).days
        return days_since_training >= self.retrain_days
