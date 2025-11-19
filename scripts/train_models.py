"""
ML Model Training Script

This script trains the XGBoost, LightGBM, and RandomForest models
used by the ML Prediction strategy.

Usage:
    python scripts/train_models.py

The script will:
1. Download historical data for training stocks
2. Calculate technical indicators and fundamentals
3. Engineer 60+ features
4. Train ensemble models
5. Validate performance
6. Save models to disk

Training takes ~15-30 minutes depending on your computer.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import yfinance as yf
from tqdm import tqdm

from stock_analyzer.ml.feature_engineer import FeatureEngineer
from stock_analyzer.ml.predictor import StockPredictor
from stock_analyzer.services.analyzer import StockAnalyzer
from stock_analyzer.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Handles the entire model training pipeline."""

    def __init__(self, output_dir: str = "models"):
        """
        Initialize trainer.

        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.feature_engineer = FeatureEngineer()
        self.predictor = StockPredictor()
        self.analyzer = StockAnalyzer()

        logger.info(f"Model trainer initialized. Models will be saved to: {self.output_dir}")

    def get_training_tickers(self) -> List[str]:
        """
        Get list of tickers for training.

        Uses liquid, large-cap stocks from various sectors for diversity.
        """
        # Top liquid stocks across sectors for good training data
        tickers = [
            # Technology
            "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CSCO",
            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "LLY", "BMY",
            # Financials
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP",
            # Consumer
            "AMZN", "TSLA", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT",
            # Industrials
            "BA", "CAT", "GE", "UPS", "HON", "MMM", "LMT", "RTX",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
            # Utilities & Staples
            "NEE", "DUK", "SO", "PG", "KO", "PEP", "WMT", "COST",
            # Communications
            "T", "VZ", "DIS", "CMCSA", "NFLX", "TMUS",
        ]

        logger.info(f"Using {len(tickers)} stocks for training")
        return tickers

    async def collect_training_data(
        self,
        tickers: List[str],
        years: int = 3
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Collect historical data and prepare training dataset.

        Args:
            tickers: List of stock tickers
            years: Years of historical data to use

        Returns:
            (features_df, direction_labels, return_labels) tuple
        """
        logger.info(f"Collecting {years} years of data for {len(tickers)} stocks...")

        all_features = []
        all_direction_labels = []
        all_return_labels = []

        # Progress bar
        pbar = tqdm(tickers, desc="Downloading data")

        for ticker in pbar:
            pbar.set_description(f"Processing {ticker}")

            try:
                # Get historical data
                stock = yf.Ticker(ticker)
                price_data = stock.history(period=f"{years}y")

                if len(price_data) < 100:
                    logger.warning(f"Insufficient data for {ticker}, skipping")
                    continue

                # Get fundamentals and info
                try:
                    analysis = await self.analyzer.analyze(ticker)
                    fundamentals = analysis.fundamentals
                    technical_indicators = analysis.technical_indicators
                    stock_info = analysis.stock_info
                except Exception as e:
                    logger.warning(f"Failed to get analysis for {ticker}: {e}")
                    fundamentals = None
                    technical_indicators = None
                    stock_info = None

                # Generate features for each time period
                # We'll use a sliding window approach
                for i in range(60, len(price_data) - 21):  # Need 60 days history, predict 21 days forward
                    window = price_data.iloc[:i+1]

                    # Engineer features
                    features = self.feature_engineer.engineer_features(
                        price_data=window,
                        fundamentals=fundamentals,
                        technical_indicators=technical_indicators,
                        stock_info=stock_info
                    )

                    if features is None or features.empty:
                        continue

                    # Calculate forward return (21 trading days ≈ 1 month)
                    current_price = window['Close'].iloc[-1]
                    future_price = price_data['Close'].iloc[i + 21]
                    forward_return = (future_price / current_price - 1)

                    # Labels
                    direction = 1 if forward_return > 0 else 0  # Binary classification

                    all_features.append(features)
                    all_direction_labels.append(direction)
                    all_return_labels.append(forward_return)

            except Exception as e:
                logger.warning(f"Error processing {ticker}: {e}")
                continue

        # Combine all data
        if not all_features:
            raise ValueError("No training data collected! Check your internet connection.")

        features_df = pd.concat(all_features, ignore_index=True)
        direction_series = pd.Series(all_direction_labels)
        returns_series = pd.Series(all_return_labels)

        logger.info(f"Collected {len(features_df)} training samples")
        logger.info(f"Positive direction: {direction_series.mean():.1%}")
        logger.info(f"Average forward return: {returns_series.mean():.2%}")

        return features_df, direction_series, returns_series

    def train_models(
        self,
        features: pd.DataFrame,
        direction_labels: pd.Series,
        return_labels: pd.Series
    ):
        """
        Train the ensemble models.

        Args:
            features: Feature dataframe
            direction_labels: Direction classification labels (0/1)
            return_labels: Return regression labels (continuous)
        """
        logger.info("Training models...")

        # Train the predictor
        metrics = self.predictor.train(
            features=features,
            direction_labels=direction_labels,
            return_labels=return_labels,
            test_size=0.2,
            random_state=42
        )

        # Log metrics
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*50)
        logger.info("\nDirection Classification Metrics:")
        logger.info(f"  Accuracy:  {metrics['direction_accuracy']:.2%}")
        logger.info(f"  Precision: {metrics['direction_precision']:.2%}")
        logger.info(f"  Recall:    {metrics['direction_recall']:.2%}")
        logger.info(f"  F1 Score:  {metrics['direction_f1']:.2%}")

        logger.info("\nReturn Prediction Metrics:")
        logger.info(f"  MAE:  {metrics['return_mae']:.4f}")
        logger.info(f"  RMSE: {metrics['return_rmse']:.4f}")
        logger.info(f"  R²:   {metrics['return_r2']:.4f}")
        logger.info("="*50 + "\n")

        return metrics

    def save_models(self):
        """Save trained models to disk."""
        logger.info(f"Saving models to {self.output_dir}...")

        self.predictor.save_models(str(self.output_dir))

        logger.info("✓ Models saved successfully!")

    async def run_training_pipeline(self, years: int = 3):
        """
        Run the complete training pipeline.

        Args:
            years: Years of historical data to use (default: 3)
        """
        print("\n" + "="*70)
        print("ML MODEL TRAINING PIPELINE")
        print("="*70)
        print(f"\nThis will train XGBoost, LightGBM, and RandomForest models")
        print(f"using {years} years of historical data.")
        print(f"\nEstimated time: 15-30 minutes")
        print("="*70 + "\n")

        # Step 1: Get tickers
        tickers = self.get_training_tickers()

        # Step 2: Collect data
        print("\n[1/3] Collecting historical data...")
        features, direction_labels, return_labels = await self.collect_training_data(
            tickers=tickers,
            years=years
        )

        # Step 3: Train models
        print("\n[2/3] Training models...")
        metrics = self.train_models(features, direction_labels, return_labels)

        # Step 4: Save models
        print("\n[3/3] Saving models...")
        self.save_models()

        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)
        print("\nYour models are ready to use!")
        print("Restart your Streamlit app to use the trained ML strategy.")
        print("="*70 + "\n")

        return metrics


async def main():
    """Main entry point for training script."""
    try:
        trainer = ModelTrainer(output_dir="models")
        await trainer.run_training_pipeline(years=3)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ Training failed: {e}")
        print("Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    # Run the training pipeline
    asyncio.run(main())
