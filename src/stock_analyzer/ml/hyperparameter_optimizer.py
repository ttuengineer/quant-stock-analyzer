"""
Hyperparameter Optimization for ML Models using Optuna.

This module uses Bayesian optimization to find optimal hyperparameters
for XGBoost, LightGBM, and RandomForest models.

Expected improvements:
- AUC: 0.649 -> 0.68-0.70 (+3-5%)
- Sharpe: 0.52 -> 0.60-0.65 (+15-25%)
- More stable predictions across different market regimes
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """
    Optimizes hyperparameters for stock prediction models.

    Uses Optuna with Tree-structured Parzen Estimator (TPE) for
    efficient Bayesian optimization.
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_splits: int = 3,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize optimizer.

        Args:
            n_trials: Number of optimization trials per model
            cv_splits: Number of time-series cross-validation splits
            random_state: Random seed
            verbose: Print progress
        """
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.verbose = verbose

        # Best parameters found
        self.best_params = {
            'xgboost_clf': None,
            'xgboost_reg': None,
            'lightgbm_clf': None,
            'lightgbm_reg': None,
            'random_forest_clf': None,
            'random_forest_reg': None
        }

    def optimize_all(
        self,
        X: np.ndarray,
        y_classification: np.ndarray,
        y_regression: np.ndarray
    ) -> Dict:
        """
        Optimize hyperparameters for all models.

        Args:
            X: Feature matrix
            y_classification: Binary labels (0/1)
            y_regression: Continuous returns

        Returns:
            Dictionary with best parameters for each model
        """
        print("=" * 70)
        print("HYPERPARAMETER OPTIMIZATION")
        print("=" * 70)
        print(f"Trials per model: {self.n_trials}")
        print(f"CV splits: {self.cv_splits}")
        print(f"Samples: {len(X)}")
        print("")

        # Optimize each model
        self.best_params['xgboost_clf'] = self._optimize_xgboost_clf(X, y_classification)
        self.best_params['xgboost_reg'] = self._optimize_xgboost_reg(X, y_regression)

        self.best_params['lightgbm_clf'] = self._optimize_lightgbm_clf(X, y_classification)
        self.best_params['lightgbm_reg'] = self._optimize_lightgbm_reg(X, y_regression)

        self.best_params['random_forest_clf'] = self._optimize_rf_clf(X, y_classification)
        self.best_params['random_forest_reg'] = self._optimize_rf_reg(X, y_regression)

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 70)

        return self.best_params

    def _optimize_xgboost_clf(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize XGBoost classifier."""
        print(f"\n[1/6] Optimizing XGBoost Classifier...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': self.random_state,
                'n_jobs': -1
            }

            return self._cv_score_classification(X, y, xgb.XGBClassifier, params)

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)

        best_score = study.best_value
        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'n_jobs': -1
        })

        print(f"  Best AUC: {best_score:.4f}")
        print(f"  Best params: n_estimators={best_params['n_estimators']}, "
              f"max_depth={best_params['max_depth']}, lr={best_params['learning_rate']:.4f}")

        return best_params

    def _optimize_xgboost_reg(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize XGBoost regressor."""
        print(f"\n[2/6] Optimizing XGBoost Regressor...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'objective': 'reg:squarederror',
                'random_state': self.random_state,
                'n_jobs': -1
            }

            return self._cv_score_regression(X, y, xgb.XGBRegressor, params)

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)

        best_score = study.best_value
        best_params = study.best_params
        best_params.update({
            'objective': 'reg:squarederror',
            'random_state': self.random_state,
            'n_jobs': -1
        })

        print(f"  Best RMSE: {best_score:.4f}")
        print(f"  Best params: n_estimators={best_params['n_estimators']}, "
              f"max_depth={best_params['max_depth']}, lr={best_params['learning_rate']:.4f}")

        return best_params

    def _optimize_lightgbm_clf(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize LightGBM classifier."""
        print(f"\n[3/6] Optimizing LightGBM Classifier...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'objective': 'binary',
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }

            return self._cv_score_classification(X, y, lgb.LGBMClassifier, params)

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)

        best_score = study.best_value
        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        })

        print(f"  Best AUC: {best_score:.4f}")
        print(f"  Best params: n_estimators={best_params['n_estimators']}, "
              f"num_leaves={best_params['num_leaves']}, lr={best_params['learning_rate']:.4f}")

        return best_params

    def _optimize_lightgbm_reg(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize LightGBM regressor."""
        print(f"\n[4/6] Optimizing LightGBM Regressor...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'objective': 'regression',
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }

            return self._cv_score_regression(X, y, lgb.LGBMRegressor, params)

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)

        best_score = study.best_value
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        })

        print(f"  Best RMSE: {best_score:.4f}")
        print(f"  Best params: n_estimators={best_params['n_estimators']}, "
              f"num_leaves={best_params['num_leaves']}, lr={best_params['learning_rate']:.4f}")

        return best_params

    def _optimize_rf_clf(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize Random Forest classifier."""
        print(f"\n[5/6] Optimizing Random Forest Classifier...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state,
                'n_jobs': -1
            }

            return self._cv_score_classification(X, y, RandomForestClassifier, params)

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)

        best_score = study.best_value
        best_params = study.best_params
        best_params.update({
            'random_state': self.random_state,
            'n_jobs': -1
        })

        print(f"  Best AUC: {best_score:.4f}")
        print(f"  Best params: n_estimators={best_params['n_estimators']}, "
              f"max_depth={best_params['max_depth']}, "
              f"max_features={best_params['max_features']}")

        return best_params

    def _optimize_rf_reg(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize Random Forest regressor."""
        print(f"\n[6/6] Optimizing Random Forest Regressor...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state,
                'n_jobs': -1
            }

            return self._cv_score_regression(X, y, RandomForestRegressor, params)

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)

        best_score = study.best_value
        best_params = study.best_params
        best_params.update({
            'random_state': self.random_state,
            'n_jobs': -1
        })

        print(f"  Best RMSE: {best_score:.4f}")
        print(f"  Best params: n_estimators={best_params['n_estimators']}, "
              f"max_depth={best_params['max_depth']}, "
              f"max_features={best_params['max_features']}")

        return best_params

    def _cv_score_classification(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class,
        params: Dict
    ) -> float:
        """Cross-validation score for classification (AUC)."""
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_class(**params)
            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)

        return np.mean(scores)

    def _cv_score_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class,
        params: Dict
    ) -> float:
        """Cross-validation score for regression (RMSE)."""
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_class(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)

        return np.mean(scores)
