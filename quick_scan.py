from stock_analyzer import StockAnalyzer
import sys

# Quick version that limits the number of stocks
analyzer = StockAnalyzer()

# Get number of stocks to analyze from command line, default 100
num_stocks = int(sys.argv[1]) if len(sys.argv) > 1 else 100

print(f"Quick scan of top {num_stocks} stocks")
print("=" * 50)

# Limit the stocks
analyzer.sp500_tickers = analyzer.sp500_tickers[:num_stocks]

# Run analysis
results = analyzer.analyze_all_stocks(top_n=30)

print(results.to_string(index=False))
results.to_csv('quick_scan_results.csv', index=False)
print(f"\nResults saved to quick_scan_results.csv")