"""
Script to collect historical data and populate the database.

Run this first to download 10 years of price/fundamental data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_analyzer.database import Database
from stock_analyzer.database.data_collector import DataCollector


def main():
    """Collect all historical data."""
    print("=" * 70)
    print("STOCK DATA COLLECTION")
    print("=" * 70)
    print("\nThis will download ~10 years of data for S&P 500 stocks.")
    print("Estimated time: 10-15 minutes")
    print("\nPress Ctrl+C to cancel\n")

    # Initialize database
    print("Initializing database...")
    db = Database(db_path="data/stocks.db", use_supabase=False)

    # Create collector
    collector = DataCollector(db)

    # Collect all data
    try:
        stats = collector.collect_all(
            start_date="2015-01-01",
            use_sp500=True
        )

        print("\n" + "=" * 70)
        print("SUCCESS! Data collection complete")
        print("=" * 70)
        print("\nYou can now run feature engineering:")
        print("  python scripts/engineer_features.py")

    except KeyboardInterrupt:
        print("\n\nCollection cancelled by user")
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
