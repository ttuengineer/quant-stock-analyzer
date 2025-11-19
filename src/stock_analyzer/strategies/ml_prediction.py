"""
Machine Learning Prediction Strategy

Uses ensemble ML models (XGBoost, LightGBM, RandomForest) to predict
stock returns and generate scores.

Combines:
- 60+ engineered features (price, technical, volume, fundamental, regime)
- Ensemble predictions for direction and magnitude
- Model confidence weighting
- Adaptive scoring based on prediction quality

Academic Foundation:
- Machine learning has shown superior performance in predicting stock returns
  compared to traditional factor models (Gu, Kelly & Xiu 2020)
- Ensemble methods reduce overfitting and improve generalization
- Feature engineering from technical and fundamental data captures
  non-linear relationships that linear models miss

References:
- Gu, Kelly & Xiu (2020): "Empirical Asset Pricing via Machine Learning"
- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"
"""

from decimal import Decimal
from typing import Optional
import pandas as pd

from .base import ScoringStrategy
from ..models.domain import Fundamentals, TechnicalIndicators, StockInfo
from ..ml.feature_engineer import FeatureEngineer
from ..ml.predictor import StockPredictor
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MLPredictionStrategy(ScoringStrategy):
    """
    ML-based scoring strategy using ensemble predictions.

    Uses XGBoost, LightGBM, and RandomForest to predict:
    - Direction: Probability of positive return
    - Magnitude: Expected return percentage
    - Confidence: Model agreement

    Score components:
    - Direction probability: 40%
    - Return magnitude: 35%
    - Model confidence: 25%
    """

    def __init__(self, weight: float = 1.0, enable_training: bool = False):
        """
        Initialize ML prediction strategy.

        Args:
            weight: Strategy weight in composite score
            enable_training: If True, collect data for periodic retraining
        """
        super().__init__(name="ML Prediction", weight=weight)

        self.feature_engineer = FeatureEngineer()
        self.predictor = StockPredictor()
        self.enable_training = enable_training

        # Training data buffer (for periodic retraining)
        self.training_buffer = []

        logger.info("Initialized ML prediction strategy")

        # Check if models need retraining
        if self.predictor.needs_retraining():
            logger.warning(
                "ML models need retraining. Predictions may be suboptimal. "
                "Enable training mode and accumulate data for retraining."
            )

    def calculate_score(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
        **kwargs
    ) -> Decimal:
        """
        Calculate ML-based score.

        Returns:
            Score from 0-100 based on ML predictions
        """
        if price_data is None or price_data.empty or len(price_data) < 60:
            logger.warning("Insufficient data for ML prediction")
            return Decimal("50.0")

        try:
            # Engineer features
            features = self.feature_engineer.engineer_features(
                price_data=price_data,
                fundamentals=fundamentals,
                technical_indicators=technical_indicators,
                stock_info=stock_info
            )

            if features is None or features.empty:
                logger.warning("Feature engineering failed")
                return Decimal("50.0")

            # Get ML prediction
            prediction = self.predictor.predict(features, return_components=False)

            # Calculate score from prediction
            score = self._prediction_to_score(prediction)

            # Optionally collect training data
            if self.enable_training:
                self._collect_training_data(features, price_data)

            logger.debug(
                f"ML Score: {score:.1f} - Prob: {prediction['direction_prob']:.3f}, "
                f"Return: {prediction['predicted_return']:.2f}%, "
                f"Confidence: {prediction['confidence']:.3f}"
            )

            return Decimal(str(score))

        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return Decimal("50.0")

    def _prediction_to_score(self, prediction: dict) -> float:
        """
        Convert ML prediction to 0-100 score.

        Components:
        1. Direction probability (40%): 0.5 = 50 points, 1.0 = 100 points
        2. Return magnitude (35%): Maps predicted return to score
        3. Model confidence (25%): Weights the prediction reliability
        """
        score = 0.0

        # Component 1: Direction probability (40 points)
        # Scale from [0.0-1.0] to [0-100], centered at 0.5=50
        direction_prob = prediction["direction_prob"]
        direction_score = (direction_prob - 0.5) * 200  # Maps [0.5-1.0] to [0-100]
        score += max(0, min(100, direction_score)) * 0.40

        # Component 2: Return magnitude (35 points)
        # Map predicted return to score (e.g., +5% = 100 points)
        predicted_return = prediction["predicted_return"]
        return_score = self._map_return_to_score(predicted_return)
        score += return_score * 0.35

        # Component 3: Model confidence (25 points)
        # High confidence boosts score, low confidence dampens it
        confidence = prediction["confidence"]
        confidence_score = confidence * 100
        score += confidence_score * 0.25

        return min(100.0, max(0.0, score))

    def _map_return_to_score(self, predicted_return: float) -> float:
        """
        Map predicted return to 0-100 score.

        Mapping:
        - >= +5%: 100 points
        - +3% to +5%: 80-100 points
        - +1% to +3%: 60-80 points
        - 0% to +1%: 50-60 points
        - 0% to -1%: 40-50 points
        - -1% to -3%: 20-40 points
        - -3% to -5%: 0-20 points
        - <= -5%: 0 points
        """
        if predicted_return >= 5.0:
            return 100.0
        elif predicted_return >= 3.0:
            return 80.0 + (predicted_return - 3.0) / 2.0 * 20.0
        elif predicted_return >= 1.0:
            return 60.0 + (predicted_return - 1.0) / 2.0 * 20.0
        elif predicted_return >= 0.0:
            return 50.0 + predicted_return * 10.0
        elif predicted_return >= -1.0:
            return 40.0 + (predicted_return + 1.0) * 10.0
        elif predicted_return >= -3.0:
            return 20.0 + (predicted_return + 3.0) / 2.0 * 20.0
        elif predicted_return >= -5.0:
            return 0.0 + (predicted_return + 5.0) / 2.0 * 20.0
        else:
            return 0.0

    def _collect_training_data(self, features: pd.DataFrame, price_data: pd.DataFrame):
        """
        Collect features and future returns for model retraining.

        This builds a dataset of (features, actual_returns) that can be used
        to retrain models periodically.
        """
        try:
            # Calculate actual forward returns (will be filled in later)
            # In production, you'd store this and update with actual returns later
            current_price = price_data['Close'].iloc[-1]

            training_sample = {
                "features": features.copy(),
                "current_price": current_price,
                "timestamp": pd.Timestamp.now()
            }

            self.training_buffer.append(training_sample)

            # Limit buffer size (e.g., 10,000 samples)
            if len(self.training_buffer) > 10000:
                self.training_buffer.pop(0)

            logger.debug(f"Training buffer: {len(self.training_buffer)} samples")

        except Exception as e:
            logger.warning(f"Error collecting training data: {e}")

    def retrain_models(self, training_data: pd.DataFrame):
        """
        Retrain ML models with new data.

        Args:
            training_data: DataFrame with features and forward_return_5d column

        This should be called periodically (e.g., monthly) with accumulated data.
        """
        logger.info("Retraining ML models...")

        try:
            # Ensure feature names are set
            if not self.predictor.feature_names:
                self.predictor.feature_names = self.feature_engineer.feature_names

            # Train models
            results = self.predictor.train(training_data)

            logger.info(
                f"Retraining complete - Accuracy: {results.get('classification_accuracy', 0):.3f}, "
                f"MAE: {results.get('regression_mae', 0):.3f}%"
            )

            return results

        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return {}

    def get_feature_importance(self, top_n: int = 20) -> dict:
        """Get top N most important features from ML models."""
        return self.predictor.get_feature_importance(top_n)

    def get_model_info(self) -> dict:
        """Get information about trained models."""
        return {
            "models_ready": self.predictor._models_ready(),
            "last_trained": self.predictor.last_trained,
            "needs_retraining": self.predictor.needs_retraining(),
            "n_features": len(self.predictor.feature_names),
            "ensemble_weights": self.predictor.ensemble_weights,
            "training_buffer_size": len(self.training_buffer) if self.enable_training else 0
        }
