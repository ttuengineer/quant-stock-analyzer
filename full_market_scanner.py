import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta
from typing import Dict, List, Tuple
import warnings
import time
warnings.filterwarnings('ignore')

class FullMarketScanner:
    """Scans entire stock markets for opportunities"""

    def get_all_stocks(self, markets: List[str] = ['sp500', 'nasdaq100']) -> List[str]:
        """Get ticker symbols from multiple markets"""
        all_tickers = []

        for market in markets:
            if market == 'sp500':
                tickers = self._get_sp500_tickers()
            elif market == 'nasdaq100':
                tickers = self._get_nasdaq100_tickers()
            elif market == 'russell2000':
                tickers = self._get_russell2000_tickers()
            elif market == 'all_us':
                tickers = self._get_all_us_stocks()
            else:
                tickers = []

            all_tickers.extend(tickers)

        # Remove duplicates
        return list(set(all_tickers))

    def _get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 stocks"""
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return table['Symbol'].str.replace('.', '-').tolist()
        except:
            return []

    def _get_nasdaq100_tickers(self) -> List[str]:
        """Get NASDAQ 100 stocks"""
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[2]
            return table['Ticker'].tolist()
        except:
            return []

    def _get_russell2000_tickers(self) -> List[str]:
        """Get Russell 2000 stocks (small cap)"""
        # Sample of Russell 2000 ETF holdings
        return ['SMCI', 'CHTR', 'APP', 'CVNA', 'PLUG', 'RIOT', 'MARA', 'LCID',
                'RIVN', 'FSR', 'NKLA', 'OPEN', 'SOFI', 'HOOD', 'AFRM', 'UPST',
                'COIN', 'RBLX', 'SNOW', 'PLTR', 'PATH', 'U', 'DDOG', 'NET',
                'CRWD', 'ZS', 'OKTA', 'TWLO', 'DOCU', 'ZM', 'ROKU', 'PINS']

    def _get_all_us_stocks(self) -> List[str]:
        """Get a broader list of US stocks"""
        # This would normally connect to a more comprehensive source
        # For now, combining multiple lists
        popular_stocks = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'TSM', 'AVGO', 'ORCL',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'CVS', 'LLY', 'MRK',
            # Financials
            'BRK-B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'SCHW', 'AXP', 'BLK',
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'DIS', 'SBUX',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
            # Growth Stocks
            'SHOP', 'SQ', 'SNAP', 'SPOT', 'ABNB', 'UBER', 'LYFT', 'DASH', 'DBX', 'TWTR',
            # Semiconductors
            'AMD', 'INTC', 'MU', 'QCOM', 'NXPI', 'MRVL', 'AMAT', 'LRCX', 'KLAC', 'ASML',
            # Industrials
            'BA', 'CAT', 'GE', 'MMM', 'UPS', 'RTX', 'LMT', 'HON', 'DE', 'UNP'
        ]
        return popular_stocks

    def quick_scan(self, ticker: str) -> Dict:
        """Quick analysis of a single stock"""
        try:
            stock = yf.Ticker(ticker)

            # Get recent price data
            df = stock.history(period='3mo')
            if df.empty:
                return None

            # Calculate key metrics
            current_price = df['Close'].iloc[-1]
            change_1w = ((df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100) if len(df) > 5 else 0
            change_1m = ((df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) * 100) if len(df) > 22 else 0

            # Calculate RSI
            rsi = ta.rsi(df['Close'], length=14)
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50

            # Volume spike
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1] if len(df) > 20 else df['Volume'].mean()
            volume_ratio = df['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1

            # Get info for fundamentals
            info = stock.info

            return {
                'ticker': ticker,
                'price': current_price,
                'change_1w': change_1w,
                'change_1m': change_1m,
                'rsi': current_rsi,
                'volume_ratio': volume_ratio,
                'pe_ratio': info.get('trailingPE'),
                'market_cap': info.get('marketCap'),
                'name': info.get('longName', ticker)
            }
        except:
            return None

    def scan_markets(self, markets=['sp500'], filters=None):
        """Scan multiple markets with filters"""
        print(f"Scanning markets: {markets}")
        print("This will take several minutes...")
        print("=" * 50)

        # Get all tickers
        all_tickers = self.get_all_stocks(markets)
        print(f"Total stocks to scan: {len(all_tickers)}")

        results = []
        batch_size = 10

        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i:i+batch_size]
            print(f"Progress: {i}/{len(all_tickers)} stocks scanned...")

            for ticker in batch:
                data = self.quick_scan(ticker)
                if data:
                    results.append(data)
                time.sleep(0.1)  # Rate limiting

        # Convert to DataFrame
        df = pd.DataFrame(results)

        if df.empty:
            return df

        # Apply filters if provided
        if filters:
            if 'min_change_1m' in filters:
                df = df[df['change_1m'] >= filters['min_change_1m']]
            if 'max_pe' in filters and 'pe_ratio' in df.columns:
                df = df[(df['pe_ratio'].isna()) | (df['pe_ratio'] <= filters['max_pe'])]
            if 'min_volume_ratio' in filters:
                df = df[df['volume_ratio'] >= filters['min_volume_ratio']]
            if 'rsi_range' in filters:
                df = df[(df['rsi'] >= filters['rsi_range'][0]) & (df['rsi'] <= filters['rsi_range'][1])]

        # Sort by 1-month change
        df = df.sort_values('change_1m', ascending=False)

        return df

def main():
    scanner = FullMarketScanner()

    print("Full Market Scanner")
    print("=" * 50)
    print("\nChoose markets to scan:")
    print("1. S&P 500 only (500 stocks)")
    print("2. NASDAQ 100 (100 tech stocks)")
    print("3. Russell 2000 (small caps)")
    print("4. All US stocks (comprehensive)")
    print("5. Everything (S&P 500 + NASDAQ 100)")

    choice = input("\nEnter choice (1-5): ")

    markets = {
        '1': ['sp500'],
        '2': ['nasdaq100'],
        '3': ['russell2000'],
        '4': ['all_us'],
        '5': ['sp500', 'nasdaq100']
    }.get(choice, ['sp500'])

    print("\nApply filters? (y/n): ", end='')
    if input().lower() == 'y':
        filters = {
            'min_change_1m': float(input("Minimum 1-month gain % (e.g., 10): ") or 0),
            'max_pe': float(input("Maximum P/E ratio (e.g., 30): ") or 1000),
            'min_volume_ratio': float(input("Minimum volume spike (e.g., 1.5): ") or 0),
            'rsi_range': (30, 70)  # Default RSI range
        }
    else:
        filters = None

    # Scan markets
    results = scanner.scan_markets(markets, filters)

    if not results.empty:
        print("\n" + "=" * 50)
        print("TOP OPPORTUNITIES FOUND:")
        print("=" * 50)

        # Display top 30
        top_picks = results.head(30)

        # Format for display
        display_cols = ['ticker', 'name', 'price', 'change_1w', 'change_1m', 'rsi', 'volume_ratio', 'pe_ratio']
        display_df = top_picks[display_cols].copy()

        # Format columns
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
        display_df['change_1w'] = display_df['change_1w'].apply(lambda x: f"{x:.1f}%")
        display_df['change_1m'] = display_df['change_1m'].apply(lambda x: f"{x:.1f}%")
        display_df['rsi'] = display_df['rsi'].apply(lambda x: f"{x:.1f}")
        display_df['volume_ratio'] = display_df['volume_ratio'].apply(lambda x: f"{x:.1f}x")
        display_df['pe_ratio'] = display_df['pe_ratio'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

        print(display_df.to_string(index=False))

        # Save results
        results.to_csv('full_market_scan.csv', index=False)
        print(f"\n✅ Full results saved to full_market_scan.csv ({len(results)} stocks)")

        # Highlight special opportunities
        print("\n" + "=" * 50)
        print("🚀 SPECIAL ALERTS:")

        # Momentum plays
        momentum = results[results['change_1m'] > 20].head(5)
        if not momentum.empty:
            print("\n📈 STRONG MOMENTUM (>20% monthly gain):")
            for _, stock in momentum.iterrows():
                print(f"  • {stock['ticker']}: {stock['change_1m']:.1f}% gain")

        # Oversold opportunities
        oversold = results[results['rsi'] < 35].head(5)
        if not oversold.empty:
            print("\n💎 OVERSOLD GEMS (RSI < 35):")
            for _, stock in oversold.iterrows():
                print(f"  • {stock['ticker']}: RSI {stock['rsi']:.1f}")

        # Volume spikes
        volume_spikes = results[results['volume_ratio'] > 2].head(5)
        if not volume_spikes.empty:
            print("\n🔥 UNUSUAL VOLUME (>2x average):")
            for _, stock in volume_spikes.iterrows():
                print(f"  • {stock['ticker']}: {stock['volume_ratio']:.1f}x normal volume")
    else:
        print("\nNo stocks found matching criteria.")

if __name__ == "__main__":
    main()