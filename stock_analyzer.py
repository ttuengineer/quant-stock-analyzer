import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self):
        self.sp500_tickers = self._get_sp500_tickers()

    def _get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 ticker symbols"""
        import os

        # Try loading from local file first
        if os.path.exists('sp500_tickers.txt'):
            try:
                with open('sp500_tickers.txt', 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(tickers)} stocks from sp500_tickers.txt")
                return tickers
            except:
                pass

        # Skip Wikipedia (often blocked) and use local list
        print("Using local S&P 500 list...")

        # Use comprehensive fallback list of actual S&P 500 stocks
        sp500_list = """MMM AOS ABT ABBV ACN ADBE AMD AES AFL A APD AKAM ALB ARE ALGN ALLE LNT ALL GOOGL GOOG MO AMZN AMCR AEE AAL AEP AXP AIG AMT AWK AMP AME AMGN APH ADI ANSS AON APA AAPL AMAT APTV ACGL ADM ANET AJG AIZ T ATO ADSK ADP AZO AVB AVY AXON BKR BALL BAC BBWI BAX BDX BBY BIO TECH BIIB BLK BX BK BA BKNG BWA BXP BSX BMY AVGO BR BRO BF-B BG CHRW CDNS CZR CPT CPB COF CAH KMX CCL CARR CTLT CAT CBOE CBRE CDW CE COR CNC CNP CDAY CF CRL SCHW CHTR CVX CMG CB CHD CI CINF CTAS CSCO C CFG CLX CME CMS KO CTSH CL CMCSA CMA CAG COP ED STZ CPRT GLW CTVA CSGP COST CTRA CCI CSX CMI CVS DHR DRI DVA DE DAL XRAY DVN DXCM FANG DLR DFS DG DLTR D DPZ DOV DOW DHI DTE DUK DD EMN ETN EBAY ECL EIX EW EA ELV LLY EMR ENPH ETR EOG EPAM EQT EFX EQIX EQR ESS EL ETSY EG EVRG ES EXC EXPE EXPD EXR XOM FFIV FDS FICO FAST FRT FDX FITB FSLR FE FIS FI FLT FMC F FTNT FTV FOXA FOX BEN FCX GRMN GPS IT GEHC GEN GNRC GD GE GIS GM GPC GILD GPN GL GS HAL HIG HAS HCA PEAK HSIC HSY HES HPE HLT HOLX HD HON HRL HST HWM HPQ HUBB HUM HBAN HII IBM IEX IDXX ITW ILMN INCY IR PODD INTC ICE IFF IP IPG INTU ISRG IVZ INVH IQV IRM JBHT JBL J JNJ JCI JPM JNPR K KVUE KDP KEY KEYS KMB KIM KMI KLAC KHC KR LH LRCX LW LVS LDOS LEN LIN LYV LKQ LMT L LOW LULU LYB MTB MRO MPC MKTX MAR MMC MLM MAS MA MTCH MKC MCD MCK MDT MRK META MET MTD MGM MCHP MU MSFT MAA MRNA MHK MOH TAP MDLZ MPWR MNST MCO MS MOS MSI MSCI NDAQ NTAP NFLX NEM NWSA NWS NEE NKE NI NDSN NSC NTRS NOC NCLH NRG NUE NVDA NVR NXPI ORLY OXY ODFL OMC ON OKE ORCL OTIS PCAR PKG PANW PARA PH PAYX PAYC PYPL PNR PEP PFE PCG PM PSX PNW PXD PNC POOL PPG PPL PFG PG PGR PLD PRU PEG PTC PSA PHM QRVO PWR QCOM DGX RL PKI PRGO REGN RF RSG RMD RVTY ROK ROL ROP ROST RCL SPGI CRM SBAC SLB STX SRE NOW SHW SPG SWKS SJM SNA SEDG SO LUV SWK SBUX STT STLD STE SYK SYF SNPS SYY TMUS TROW TTWO TPR TRGP TGT TEL TDY TFX TER TSLA TXN TXT TMO TJX TSCO TT TDG TRV TRMB TFC TYL TSN USB UDR ULTA UNP UAL UPS URI UNH UHS VLO VTR VLTO VRSN VRSK VZ VRTX VFC VTRS VICI V VMC WRB GWW WAB WBA WMT DIS WM WAT WEC WFC WELL WST WDC WRK WY WTW WYN WYNN XEL XYL YUM ZBRA ZBH ZION ZTS""".split()

        print(f"Using comprehensive S&P 500 list with {len(sp500_list)} stocks")

        # Save for next time if not already saved
        if not os.path.exists('sp500_tickers.txt'):
            with open('sp500_tickers.txt', 'w') as f:
                for ticker in sp500_list:
                    f.write(f"{ticker}\n")

        return sp500_list

    def fetch_stock_data(self, ticker: str, period: str = '6mo') -> pd.DataFrame:
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            return data
        except:
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate various technical indicators"""
        if df.empty:
            return {}

        indicators = {}

        # Price momentum
        indicators['price_change_1m'] = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) * 100 if len(df) > 22 else 0
        indicators['price_change_3m'] = (df['Close'].iloc[-1] / df['Close'].iloc[-66] - 1) * 100 if len(df) > 66 else 0

        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        indicators['above_sma20'] = df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else False
        indicators['above_sma50'] = df['Close'].iloc[-1] > df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else False

        # RSI
        rsi = ta.rsi(df['Close'], length=14)
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50

        # MACD
        macd = ta.macd(df['Close'])
        if not macd.empty:
            indicators['macd_signal'] = (macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]) if 'MACD_12_26_9' in macd.columns else False

        # Volume
        indicators['volume_ratio'] = df['Volume'].iloc[-1] / df['Volume'].rolling(window=20).mean().iloc[-1] if len(df) > 20 else 1

        # Volatility
        returns = df['Close'].pct_change()
        indicators['volatility'] = returns.std() * np.sqrt(252) * 100  # Annualized volatility

        # 52-week high/low
        week52_high = df['High'].rolling(window=252).max().iloc[-1] if len(df) > 252 else df['High'].max()
        week52_low = df['Low'].rolling(window=252).min().iloc[-1] if len(df) > 252 else df['Low'].min()
        indicators['pct_from_52w_high'] = ((df['Close'].iloc[-1] - week52_high) / week52_high) * 100
        indicators['pct_from_52w_low'] = ((df['Close'].iloc[-1] - week52_low) / week52_low) * 100

        return indicators

    def get_fundamental_data(self, ticker: str) -> Dict:
        """Fetch fundamental data for a stock"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamentals = {
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'profit_margin': info.get('profitMargins', None),
                'roe': info.get('returnOnEquity', None),
                'market_cap': info.get('marketCap', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'recommendation': info.get('recommendationKey', None),
                'analyst_rating': info.get('recommendationMean', None)
            }
            return fundamentals
        except:
            return {}

    def calculate_opportunity_score(self, ticker: str, technical: Dict, fundamental: Dict) -> float:
        """Calculate a composite opportunity score for a stock"""
        score = 0
        max_score = 0

        # Technical scoring (40% weight)
        if technical:
            # Momentum scores
            if technical.get('price_change_1m', 0) > 0:
                score += 5
            if technical.get('price_change_3m', 0) > 5:
                score += 5
            max_score += 10

            # Trend scores
            if technical.get('above_sma20', False):
                score += 3
            if technical.get('above_sma50', False):
                score += 3
            max_score += 6

            # RSI scores (best between 30-70, oversold < 30 is opportunity)
            rsi = technical.get('rsi', 50)
            if 30 <= rsi <= 70:
                score += 3
            elif rsi < 30:  # Oversold
                score += 5
            max_score += 5

            # MACD signal
            if technical.get('macd_signal', False):
                score += 3
            max_score += 3

            # Volume surge
            if technical.get('volume_ratio', 1) > 1.5:
                score += 2
            max_score += 2

            # Near 52-week low (potential bounce)
            if technical.get('pct_from_52w_low', 100) < 20:
                score += 4
            max_score += 4

        # Fundamental scoring (60% weight)
        if fundamental:
            # P/E ratio (lower is better for value)
            pe = fundamental.get('pe_ratio')
            if pe and 0 < pe < 15:
                score += 8
            elif pe and 15 <= pe < 25:
                score += 5
            max_score += 8

            # PEG ratio (< 1 is good)
            peg = fundamental.get('peg_ratio')
            if peg and 0 < peg < 1:
                score += 6
            elif peg and 1 <= peg < 1.5:
                score += 3
            max_score += 6

            # Profit margin
            margin = fundamental.get('profit_margin')
            if margin and margin > 0.15:
                score += 4
            elif margin and margin > 0.05:
                score += 2
            max_score += 4

            # ROE
            roe = fundamental.get('roe')
            if roe and roe > 0.15:
                score += 4
            max_score += 4

            # Growth
            rev_growth = fundamental.get('revenue_growth')
            if rev_growth and rev_growth > 0.10:
                score += 4
            max_score += 4

            # Analyst recommendation
            rating = fundamental.get('analyst_rating')
            if rating and rating <= 2:  # Strong buy = 1, Buy = 2
                score += 4
            elif rating and rating <= 3:  # Hold = 3
                score += 2
            max_score += 4

        # Normalize score to 0-100
        if max_score > 0:
            return (score / max_score) * 100
        return 0

    def analyze_all_stocks(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze all stocks and return top opportunities"""
        results = []

        print(f"Analyzing {len(self.sp500_tickers)} stocks...")

        for i, ticker in enumerate(self.sp500_tickers):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(self.sp500_tickers)}")

            try:
                # Fetch data
                df = self.fetch_stock_data(ticker)
                if df.empty:
                    continue

                # Calculate indicators
                technical = self.calculate_technical_indicators(df)
                fundamental = self.get_fundamental_data(ticker)

                # Calculate score
                score = self.calculate_opportunity_score(ticker, technical, fundamental)

                # Compile results
                result = {
                    'Ticker': ticker,
                    'Score': score,
                    'Current Price': df['Close'].iloc[-1],
                    '1M Change %': technical.get('price_change_1m', 0),
                    '3M Change %': technical.get('price_change_3m', 0),
                    'RSI': technical.get('rsi', None),
                    'Volume Ratio': technical.get('volume_ratio', 1),
                    'P/E': fundamental.get('pe_ratio', None),
                    'PEG': fundamental.get('peg_ratio', None),
                    'ROE': fundamental.get('roe', None),
                    'Profit Margin': fundamental.get('profit_margin', None),
                    'Analyst Rating': fundamental.get('analyst_rating', None),
                    '52W Low %': technical.get('pct_from_52w_low', None),
                    '52W High %': technical.get('pct_from_52w_high', None)
                }
                results.append(result)

            except Exception as e:
                continue

        # Create DataFrame and sort by score
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Score', ascending=False)

        return df_results.head(top_n)

    def get_buy_signals(self) -> pd.DataFrame:
        """Get stocks with strong buy signals"""
        df = self.analyze_all_stocks(top_n=30)

        # Filter for strong opportunities
        buy_signals = df[
            (df['Score'] > 60) &
            (df['RSI'] < 70) &  # Not overbought
            (df['1M Change %'] > -10)  # Not in freefall
        ]

        return buy_signals

def main():
    print("Stock Opportunity Analyzer")
    print("=" * 50)

    analyzer = StockAnalyzer()

    print("\nFinding top investment opportunities...")
    print("This may take a few minutes...\n")

    # Get top opportunities
    opportunities = analyzer.analyze_all_stocks(top_n=20)

    print("\nTop 20 Stock Opportunities (Scored 0-100):")
    print("=" * 50)

    # Display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)

    print(opportunities.to_string(index=False))

    # Save to CSV
    opportunities.to_csv('stock_opportunities.csv', index=False)
    print("\nResults saved to stock_opportunities.csv")

    # Get specific buy signals
    print("\n\nStrong Buy Signals (Score > 60):")
    print("=" * 50)
    buy_signals = opportunities[opportunities['Score'] > 60]
    if not buy_signals.empty:
        print(buy_signals[['Ticker', 'Score', 'Current Price', '1M Change %', 'RSI', 'P/E', 'Analyst Rating']].to_string(index=False))
    else:
        print("No strong buy signals at this time.")

if __name__ == "__main__":
    main()