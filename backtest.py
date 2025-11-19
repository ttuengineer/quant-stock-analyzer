import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stock_analyzer import StockAnalyzer
import warnings
warnings.filterwarnings('ignore')

class StrategyBacktester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.analyzer = StockAnalyzer()

    def backtest_strategy(self, start_date, end_date, rebalance_frequency='monthly'):
        """
        Backtest the opportunity scoring strategy
        """
        results = {
            'dates': [],
            'portfolio_value': [],
            'returns': [],
            'trades': []
        }

        current_capital = self.initial_capital
        current_positions = {}

        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(start_date, end_date, rebalance_frequency)

        print(f"Backtesting from {start_date} to {end_date}")
        print(f"Rebalancing {rebalance_frequency}")
        print("=" * 50)

        for i, rebalance_date in enumerate(rebalance_dates):
            print(f"\nRebalancing on {rebalance_date.strftime('%Y-%m-%d')}")

            # Get top stocks based on scoring
            top_stocks = self._get_top_stocks_historical(rebalance_date, n_stocks=5)

            if not top_stocks:
                continue

            # Calculate equal weight for each position
            position_size = current_capital / len(top_stocks)

            # Close existing positions
            for ticker in list(current_positions.keys()):
                if ticker not in top_stocks:
                    # Sell position
                    sell_price = self._get_stock_price(ticker, rebalance_date)
                    if sell_price:
                        shares = current_positions[ticker]['shares']
                        current_capital += shares * sell_price

                        profit = (sell_price - current_positions[ticker]['buy_price']) * shares
                        results['trades'].append({
                            'date': rebalance_date,
                            'ticker': ticker,
                            'action': 'SELL',
                            'price': sell_price,
                            'shares': shares,
                            'profit': profit
                        })

                        del current_positions[ticker]

            # Open new positions
            for ticker in top_stocks:
                if ticker not in current_positions:
                    buy_price = self._get_stock_price(ticker, rebalance_date)
                    if buy_price and buy_price > 0:
                        shares = position_size / buy_price
                        current_positions[ticker] = {
                            'shares': shares,
                            'buy_price': buy_price
                        }
                        current_capital -= position_size

                        results['trades'].append({
                            'date': rebalance_date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'price': buy_price,
                            'shares': shares
                        })

            # Calculate portfolio value
            portfolio_value = current_capital
            for ticker, position in current_positions.items():
                current_price = self._get_stock_price(ticker, rebalance_date)
                if current_price:
                    portfolio_value += position['shares'] * current_price

            results['dates'].append(rebalance_date)
            results['portfolio_value'].append(portfolio_value)
            results['returns'].append((portfolio_value / self.initial_capital - 1) * 100)

            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Return: {results['returns'][-1]:.2f}%")
            print(f"Holdings: {list(current_positions.keys())}")

        return results

    def _generate_rebalance_dates(self, start_date, end_date, frequency):
        """Generate rebalancing dates"""
        dates = []
        current_date = start_date

        while current_date <= end_date:
            dates.append(current_date)

            if frequency == 'weekly':
                current_date += timedelta(days=7)
            elif frequency == 'monthly':
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            elif frequency == 'quarterly':
                # Move 3 months forward
                for _ in range(3):
                    if current_date.month == 12:
                        current_date = current_date.replace(year=current_date.year + 1, month=1)
                    else:
                        current_date = current_date.replace(month=current_date.month + 1)

        return dates

    def _get_top_stocks_historical(self, date, n_stocks=5):
        """Get top stocks based on scoring at a specific date"""
        # For simplicity, using current scoring but with historical data
        # In production, you'd want to ensure point-in-time accuracy
        top_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
                      'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA']

        scored_stocks = []
        for ticker in top_tickers:
            try:
                # Get historical data up to the rebalance date
                stock = yf.Ticker(ticker)
                end = date
                start = date - timedelta(days=180)
                df = stock.history(start=start, end=end)

                if not df.empty:
                    # Calculate simple momentum score
                    returns_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) if len(df) > 22 else 0
                    returns_3m = (df['Close'].iloc[-1] / df['Close'].iloc[-66] - 1) if len(df) > 66 else 0

                    # Simple scoring based on momentum
                    score = (returns_1m * 0.3 + returns_3m * 0.7) * 100

                    scored_stocks.append((ticker, score))
            except:
                continue

        # Sort by score and return top N
        scored_stocks.sort(key=lambda x: x[1], reverse=True)
        return [ticker for ticker, _ in scored_stocks[:n_stocks]]

    def _get_stock_price(self, ticker, date):
        """Get stock price at a specific date"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=date - timedelta(days=5), end=date + timedelta(days=1))
            if not df.empty:
                # Get the closest available price
                return df['Close'].iloc[-1]
        except:
            pass
        return None

    def compare_with_benchmark(self, results, benchmark='SPY'):
        """Compare strategy performance with a benchmark"""
        if not results['dates']:
            return None

        start_date = results['dates'][0]
        end_date = results['dates'][-1]

        # Get benchmark data
        spy = yf.Ticker(benchmark)
        spy_data = spy.history(start=start_date, end=end_date)

        if spy_data.empty:
            return None

        # Calculate benchmark returns
        benchmark_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1) * 100
        strategy_return = results['returns'][-1] if results['returns'] else 0

        # Calculate Sharpe ratio (simplified)
        if len(results['returns']) > 1:
            returns_array = np.array(results['returns'])
            returns_diff = np.diff(returns_array)
            sharpe = np.mean(returns_diff) / (np.std(returns_diff) + 1e-6) * np.sqrt(12)  # Annualized
        else:
            sharpe = 0

        # Calculate max drawdown
        portfolio_values = np.array(results['portfolio_value'])
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        max_drawdown = np.min(drawdown)

        comparison = {
            'Strategy Return': f"{strategy_return:.2f}%",
            'Benchmark Return': f"{benchmark_return:.2f}%",
            'Outperformance': f"{strategy_return - benchmark_return:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Total Trades': len(results['trades']),
            'Final Value': f"${results['portfolio_value'][-1]:.2f}" if results['portfolio_value'] else "N/A"
        }

        return comparison

def main():
    print("Stock Strategy Backtester")
    print("=" * 50)

    # Initialize backtester
    backtester = StrategyBacktester(initial_capital=10000)

    # Set backtest period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year backtest

    # Run backtest
    results = backtester.backtest_strategy(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency='monthly'
    )

    # Compare with benchmark
    print("\n" + "=" * 50)
    print("Performance Summary")
    print("=" * 50)

    comparison = backtester.compare_with_benchmark(results, benchmark='SPY')

    if comparison:
        for key, value in comparison.items():
            print(f"{key}: {value}")

    # Save trade history
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv('/mnt/c/Users/dgarz/OneDrive/Desktop/Dev/stock_analyzer/backtest_trades.csv', index=False)
        print("\nTrade history saved to backtest_trades.csv")

    print("\n⚠️  Note: This is a simplified backtest. Real-world performance may vary due to:")
    print("- Transaction costs and slippage")
    print("- Market impact")
    print("- Survivorship bias")
    print("- Look-ahead bias in the scoring system")

if __name__ == "__main__":
    main()