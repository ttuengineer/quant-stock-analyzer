"""
Quick demo of the Stock Analyzer system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stock_analyzer.services.analyzer import StockAnalyzer
from stock_analyzer.data.provider_manager import ProviderManager


async def demo():
    """Demonstrate the stock analyzer."""
    print("=" * 80)
    print("STOCK ANALYZER - INSTITUTIONAL-GRADE ANALYSIS")
    print("=" * 80)

    async with ProviderManager() as provider_manager:
        analyzer = StockAnalyzer(provider_manager=provider_manager)

        # Analyze Apple
        print("\nAnalyzing AAPL (Apple Inc.)...")
        print("-" * 80)

        analysis = await analyzer.analyze("AAPL")

        print(f"\nTicker: {analysis.ticker}")
        print(f"Composite Score: {analysis.composite_score:.1f}/100")
        print(f"Signal: {analysis.signal.value.upper()}")
        print(f"Trend: {analysis.trend.value}")
        print(f"Confidence: {float(analysis.confidence)*100:.0f}%")

        if analysis.quote:
            print(f"\nCurrent Price: ${analysis.quote.price:.2f}")
            if analysis.quote.change_percent:
                print(f"Change: {analysis.quote.change_percent:.2f}%")

        if analysis.factor_scores:
            print(f"\nFactor Scores:")
            print(f"  Momentum: {analysis.factor_scores.momentum_score:.1f}/100")
            print(f"  Value: {analysis.factor_scores.value_score:.1f}/100")
            print(f"  Growth: {analysis.factor_scores.growth_score:.1f}/100")

        if analysis.technical_indicators:
            print(f"\nTechnical Indicators:")
            if analysis.technical_indicators.rsi:
                print(f"  RSI: {analysis.technical_indicators.rsi:.1f}")
            if analysis.technical_indicators.macd and analysis.technical_indicators.macd_signal:
                macd_signal = "Bullish" if float(analysis.technical_indicators.macd) > float(analysis.technical_indicators.macd_signal) else "Bearish"
                print(f"  MACD: {macd_signal}")

        if analysis.fundamentals:
            print(f"\nFundamentals:")
            if analysis.fundamentals.pe_ratio:
                print(f"  P/E Ratio: {analysis.fundamentals.pe_ratio:.2f}")
            if analysis.fundamentals.roe:
                print(f"  ROE: {float(analysis.fundamentals.roe)*100:.1f}%")
            if analysis.fundamentals.profit_margin:
                print(f"  Profit Margin: {float(analysis.fundamentals.profit_margin)*100:.1f}%")

        if analysis.key_strengths:
            print(f"\nKey Strengths:")
            for strength in analysis.key_strengths:
                print(f"  + {strength}")

        if analysis.key_risks:
            print(f"\nKey Risks:")
            for risk in analysis.key_risks:
                print(f"  - {risk}")

        # Batch analysis
        print("\n\n" + "=" * 80)
        print("BATCH ANALYSIS: Tech Giants")
        print("=" * 80)

        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        print(f"\nAnalyzing {len(tickers)} stocks...")

        analyses = await analyzer.analyze_batch(tickers)

        print(f"\n{'Ticker':<8} {'Score':<7} {'Signal':<12} {'Trend':<10} {'Price':<12}")
        print("-" * 60)

        for a in sorted(analyses, key=lambda x: x.composite_score, reverse=True):
            price = f"${float(a.quote.price):.2f}" if a.quote else "N/A"
            print(f"{a.ticker:<8} {float(a.composite_score):<7.1f} {a.signal.value:<12} {a.trend.value:<10} {price:<12}")

        print("\n" + "=" * 80)
        print("DEMO COMPLETE!")
        print("=" * 80)
        print("\nThe system is working with:")
        print("  - Live data from Yahoo Finance (no hardcoded tickers)")
        print("  - Multi-factor scoring (Momentum, Value, Growth)")
        print("  - Advanced technical indicators")
        print("  - Comprehensive fundamental analysis")
        print("  - Async batch processing")
        print("\nNext steps:")
        print("  - Add User-Agent to bypass Wikipedia blocking for S&P 500 fetching")
        print("  - Use CLI: python -m src.stock_analyzer.cli.main analyze <TICKER>")
        print("  - Add more data providers (Alpha Vantage, Polygon.io)")
        print("  - Implement ML models for price prediction")


if __name__ == "__main__":
    asyncio.run(demo())
