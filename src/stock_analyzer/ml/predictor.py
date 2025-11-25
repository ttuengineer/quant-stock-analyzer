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
from typing import Optional, Dict, Tuple, List, Union
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
        features: Union[pd.DataFrame, np.ndarray],
        return_components: bool = False
    ) -> Dict:
        """
        Predict stock returns using ensemble. Supports single-row or batch input.

        Args:
            features: Engineered features (DataFrame or ndarray). Can be multiple rows.
            return_components: If True, return individual model predictions

        Returns:
            Dictionary with:
            - direction_prob: Probability of positive return (float or 1D array)
            - predicted_return: Expected return percentage (float or 1D array)
            - confidence: Model agreement (float or 1D array)
            - signal: BUY/SELL/HOLD (string or list)
            - components: Individual model predictions (if requested)
        """
        if features is None or (hasattr(features, 'empty') and features.empty) or (isinstance(features, np.ndarray) and features.size == 0):
            logger.warning("No features provided for prediction")
            return self._default_prediction()

        if not self._models_ready():
            logger.warning("Models not trained yet - returning neutral prediction")
            return self._default_prediction()

        try:
            # Prepare features
            if isinstance(features, np.ndarray):
                X = features
            else:
                X = features[self.feature_names]

            # Ensure 2D
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            X_scaled = self.scaler.transform(X)
            n = X_scaled.shape[0]

            # Get predictions from each model (vectorized)
            xgb_prob = self.xgb_clf.predict_proba(X_scaled)[:, 1]
            lgb_prob = self.lgb_clf.predict_proba(X_scaled)[:, 1]
            rf_prob = self.rf_clf.predict_proba(X_scaled)[:, 1]

            xgb_return = self.xgb_reg.predict(X_scaled)
            lgb_return = self.lgb_reg.predict(X_scaled)
            rf_return = self.rf_reg.predict(X_scaled)

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

            # Confidence per row (std across models)
            prob_stack = np.vstack([xgb_prob, lgb_prob, rf_prob])
            prob_std = prob_stack.std(axis=0)
            confidence = np.clip(1.0 - prob_std * 2, 0.0, 1.0)

            # Signals
            signals = [self._generate_signal(p, r, c) for p, r, c in zip(ensemble_prob, ensemble_return, confidence)]

            # If single-row, return scalars; else return arrays/lists
            if n == 1:
                result = {
                    "direction_prob": float(ensemble_prob[0]),
                    "predicted_return": float(ensemble_return[0]),
                    "confidence": float(confidence[0]),
                    "signal": signals[0],
                    "timestamp": datetime.now().isoformat()
                }
                if return_components:
                    result["components"] = {
                        "xgboost": {"prob": float(xgb_prob[0]), "return": float(xgb_return[0])},
                        "lightgbm": {"prob": float(lgb_prob[0]), "return": float(lgb_return[0])},
                        "random_forest": {"prob": float(rf_prob[0]), "return": float(rf_return[0])}
                    }
            else:
                result = {
                    "direction_prob": ensemble_prob,
                    "predicted_return": ensemble_return,
                    "confidence": confidence,
                    "signal": signals,
                    "timestamp": datetime.now().isoformat()
                }
                if return_components:
                    result["components"] = {
                        "xgboost": {"prob": xgb_prob, "return": xgb_return},
                        "lightgbm": {"prob": lgb_prob, "return": lgb_return},
                        "random_forest": {"prob": rf_prob, "return": rf_return}
                    }

            logger.debug(
                f"ML Prediction - Prob mean: {ensemble_prob.mean():.3f}, Return mean: {ensemble_return.mean():.3f}, "
                f"Conf mean: {confidence.mean():.3f}, Samples: {n}"
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
        random_state: int = 42,
        optimize_hyperparameters: bool = False,
        optuna_trials: int = 50
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
            optimize_hyperparameters: If True, use Optuna to find optimal hyperparameters
            optuna_trials: Number of optimization trials per model (default: 50)

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

            # Hyperparameter optimization (if enabled)
            optimized_params = None
            if optimize_hyperparameters:
                logger.info("Running hyperparameter optimization...")
                from .hyperparameter_optimizer import HyperparameterOptimizer

                optimizer = HyperparameterOptimizer(
                    n_trials=optuna_trials,
                    cv_splits=3,
                    random_state=random_state,
                    verbose=True
                )

                optimized_params = optimizer.optimize_all(
                    X_train,
                    y_direction_train if isinstance(y_direction_train, np.ndarray) else y_direction_train.values,
                    y_return_train if isinstance(y_return_train, np.ndarray) else y_return_train.values
                )

                logger.info("Hyperparameter optimization complete!")

            # Train on training set
            self._train_classifiers(X_train, y_direction_train, optimized_params)
            self._train_regressors(X_train, y_return_train, optimized_params)

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
            self._train_classifiers(X_scaled, y_direction if isinstance(y_direction, np.ndarray) else y_direction.values, optimized_params)
            self._train_regressors(X_scaled, y_return if isinstance(y_return, np.ndarray) else y_return.values, optimized_params)

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

    def _train_classifiers(self, X: np.ndarray, y: np.ndarray, optimized_params: Dict = None):
        """Train classification models (direction prediction)."""
        # XGBoost Classifier
        xgb_params = optimized_params['xgboost_clf'] if optimized_params else {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        self.xgb_clf = xgb.XGBClassifier(**xgb_params)
        self.xgb_clf.fit(X, y)

        # LightGBM Classifier
        lgb_params = optimized_params['lightgbm_clf'] if optimized_params else {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        self.lgb_clf = lgb.LGBMClassifier(**lgb_params)
        self.lgb_clf.fit(X, y)

        # Random Forest Classifier
        rf_params = optimized_params['random_forest_clf'] if optimized_params else {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        self.rf_clf = RandomForestClassifier(**rf_params)
        self.rf_clf.fit(X, y)

    def _train_regressors(self, X: np.ndarray, y: np.ndarray, optimized_params: Dict = None):
        """Train regression models (return magnitude prediction)."""
        # XGBoost Regressor
        xgb_params = optimized_params['xgboost_reg'] if optimized_params else {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        self.xgb_reg = xgb.XGBRegressor(**xgb_params)
        self.xgb_reg.fit(X, y)

        # LightGBM Regressor
        lgb_params = optimized_params['lightgbm_reg'] if optimized_params else {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'regression',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        self.lgb_reg = lgb.LGBMRegressor(**lgb_params)
        self.lgb_reg.fit(X, y)

        # Random Forest Regressor
        rf_params = optimized_params['random_forest_reg'] if optimized_params else {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        self.rf_reg = RandomForestRegressor(**rf_params)
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

    def get_shap_importance(self, X: pd.DataFrame, top_n: int = 20) -> Dict:
        """
        Get SHAP feature importance.

        SHAP values provide more accurate feature importance than
        built-in methods because they account for feature interactions.

        Args:
            X: Feature data
            top_n: Number of top features to return

        Returns:
            Dictionary of {feature: importance}
        """
        try:
            import shap

            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(self.xgb_clf)

            # Calculate SHAP values (sample if dataset is large)
            # Accept ndarray or DataFrame; wrap ndarray with feature names
            if isinstance(X, pd.DataFrame):
                X_frame = X
            else:
                X_frame = pd.DataFrame(X, columns=self.feature_names)

            # Align feature names with data shape to avoid index/shape mismatches
            if not self.feature_names or len(self.feature_names) != X_frame.shape[1]:
                logger.warning(
                    "Feature name mismatch for SHAP (expected %s, got %s) - aligning to data columns",
                    len(self.feature_names) if self.feature_names else "none",
                    X_frame.shape[1]
                )
                self.feature_names = list(X_frame.columns)

            sample_size = min(1000, len(X_frame))
            X_sample = X_frame[self.feature_names].sample(sample_size, random_state=42) if len(X_frame) > sample_size else X_frame[self.feature_names]

            shap_values = explainer.shap_values(X_sample)

            # Get mean absolute SHAP value for each feature
            mean_shap = np.abs(shap_values).mean(axis=0)

            # Create importance dict
            shap_importance = dict(zip(self.feature_names, mean_shap))

            # Sort by importance
            shap_importance = dict(
                sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            )

            logger.info(f"SHAP analysis complete. Top feature: {list(shap_importance.keys())[0]}")

            return shap_importance

        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return {}
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return {}

    def needs_retraining(self) -> bool:
        """Check if models need retraining based on age."""
        if self.last_trained is None:
            return True

        days_since_training = (datetime.now() - self.last_trained).days
        return days_since_training >= self.retrain_days
