"""
Engineer features for ML model training.

Run after collecting data to compute all features and targets.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_analyzer.database import Database
from stock_analyzer.ml_features import FeatureEngineer


def main():
    """Engineer all features."""
    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    print("\nThis will compute features for all stocks on monthly dates.")
    print("Estimated time: 2-5 minutes\n")

    # Initialize database
    print("Initializing database...")
    db = Database(db_path="data/stocks.db", use_supabase=False)

    # Create feature engineer
    engineer = FeatureEngineer(db)

    try:
        # Engineer features (from 2015 to present)
        # Using 3-MONTH target (63 trading days) - much less noisy than 1-month!
        print("\nEngineering features with 3-MONTH prediction horizon...")
        features_df = engineer.engineer_all_features(
            start_date="2015-01-01",
            forward_window=63  # 3 months = ~63 trading days (less noisy!)
        )

        print("\n" + "=" * 70)
        print("SUCCESS! Feature engineering complete")
        print("=" * 70)

        # Show sample of features
        if not features_df.empty:
            print(f"\nGenerated {len(features_df)} training samples")
            print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")
            print(f"\nSample features:")
            print(features_df.head(3)[['ticker', 'date', 'return_3m_rank', 'dist_from_52w_high_rank', 'target_binary']])

            print("\nYou can now train the model:")
            print("  python scripts/train_model.py")
        else:
            print("\nWARNING: No features generated. Check data.")

    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
