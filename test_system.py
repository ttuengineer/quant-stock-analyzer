"""
Quick system test to verify everything works.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stock_analyzer.services.analyzer import StockAnalyzer
from stock_analyzer.data.provider_manager import ProviderManager


async def test_system():
    """Test the stock analyzer system."""
    print("=" * 80)
    print("STOCK ANALYZER SYSTEM TEST")
    print("=" * 80)

    # Test 1: Initialize provider manager
    print("\n1. Initializing provider manager...")
    async with ProviderManager() as provider_manager:
        print("   [OK] Provider manager initialized")

        # Test 2: Fetch S&P 500 tickers (live data, not hardcoded!)
        print("\n2. Fetching live S&P 500 constituents...")
        try:
            tickers = await provider_manager.fetch_sp500_tickers()
            print(f"   [OK] Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
            print(f"   Sample tickers: {', '.join(tickers[:10])}...")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            return

        # Test 3: Analyze a single stock
        print("\n3. Analyzing AAPL stock...")
        analyzer = StockAnalyzer(provider_manager=provider_manager)

        try:
            analysis = await analyzer.analyze("AAPL")
            print(f"   [OK] Analysis complete!")
            print(f"   Ticker: {analysis.ticker}")
            print(f"   Composite Score: {analysis.composite_score:.1f}/100")
            print(f"   Signal: {analysis.signal.value.upper()}")
            print(f"   Trend: {analysis.trend.value}")

            if analysis.quote:
                print(f"   Price: ${analysis.quote.price:.2f}")

            if analysis.technical_indicators and analysis.technical_indicators.rsi:
                print(f"   RSI: {analysis.technical_indicators.rsi:.1f}")

            if analysis.fundamentals and analysis.fundamentals.pe_ratio:
                print(f"   P/E Ratio: {analysis.fundamentals.pe_ratio:.1f}")

            if analysis.key_strengths:
                print(f"   Strengths:")
                for strength in analysis.key_strengths[:3]:
                    print(f"     - {strength}")

        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            return

        # Test 4: Batch analysis
        print("\n4. Testing batch analysis (AAPL, MSFT, GOOGL)...")
        try:
            analyses = await analyzer.analyze_batch(["AAPL", "MSFT", "GOOGL"])
            print(f"   [OK] Analyzed {len(analyses)} stocks")

            for a in analyses:
                print(f"   {a.ticker}: Score={a.composite_score:.1f}, Signal={a.signal.value}")

        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            return

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSystem is ready to use!")
    print("\nTry these commands:")
    print("  python -m src.stock_analyzer.cli.main analyze AAPL")
    print("  python -m src.stock_analyzer.cli.main screen --top 20")


if __name__ == "__main__":
    asyncio.run(test_system())
