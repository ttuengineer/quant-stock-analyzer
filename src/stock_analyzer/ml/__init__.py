"""
Machine Learning module for stock prediction.

Includes:
- Feature engineering (60+ technical & fundamental features)
- XGBoost, LightGBM, RandomForest models
- Ensemble predictions
- Walk-forward validation
"""

from .feature_engineer import FeatureEngineer
from .predictor import StockPredictor

__all__ = ["FeatureEngineer", "StockPredictor"]
