"""
Command-line interface for Stock Analyzer.

Provides intuitive commands for stock analysis and screening.
"""

import asyncio
import argparse
import sys
from typing import List
from decimal import Decimal

from ..services.analyzer import StockAnalyzer
from ..data.provider_manager import ProviderManager
from ..models.domain import Analysis
from ..models.enums import SignalType
from ..config import get_settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class CLI:
    """Command-line interface."""

    def __init__(self):
        self.settings = get_settings()
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Stock Analyzer - Institutional-Grade Investment Analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Analyze a single stock
  python -m stock_analyzer.cli.main analyze AAPL

  # Screen S&P 500 for top opportunities
  python -m stock_analyzer.cli.main screen --top 20

  # Find strong buy signals
  python -m stock_analyzer.cli.main screen --signal strong_buy

  # Custom stock list
  python -m stock_analyzer.cli.main analyze AAPL MSFT GOOGL TSLA
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Commands')

        # Analyze command
        analyze_parser = subparsers.add_parser(
            'analyze',
            help='Analyze one or more stocks'
        )
        analyze_parser.add_argument(
            'tickers',
            nargs='+',
            help='Stock ticker symbols (e.g., AAPL MSFT GOOGL)'
        )
        analyze_parser.add_argument(
            '--output',
            choices=['table', 'json', 'csv'],
            default='table',
            help='Output format (default: table)'
        )

        # Screen command
        screen_parser = subparsers.add_parser(
            'screen',
            help='Screen market for opportunities'
        )
        screen_parser.add_argument(
            '--universe',
            choices=['sp500', 'nasdaq', 'nyse'],
            default='sp500',
            help='Stock universe to screen (default: sp500)'
        )
        screen_parser.add_argument(
            '--top',
            type=int,
            default=20,
            help='Number of top results to show (default: 20)'
        )
        screen_parser.add_argument(
            '--signal',
            choices=['strong_buy', 'buy', 'hold', 'sell', 'strong_sell'],
            help='Filter by signal type'
        )
        screen_parser.add_argument(
            '--min-score',
            type=float,
            default=60.0,
            help='Minimum composite score (default: 60)'
        )
        screen_parser.add_argument(
            '--output',
            choices=['table', 'json', 'csv'],
            default='table',
            help='Output format (default: table)'
        )

        return parser

    async def run(self, args: List[str] = None):
        """Run CLI with given arguments."""
        parsed_args = self.parser.parse_args(args)

        if not parsed_args.command:
            self.parser.print_help()
            return

        # Initialize provider manager
        async with ProviderManager() as provider_manager:
            analyzer = StockAnalyzer(provider_manager=provider_manager)

            if parsed_args.command == 'analyze':
                await self._handle_analyze(analyzer, parsed_args)
            elif parsed_args.command == 'screen':
                await self._handle_screen(analyzer, provider_manager, parsed_args)

    async def _handle_analyze(self, analyzer: StockAnalyzer, args):
        """Handle analyze command."""
        print(f"\n{'='*80}")
        print(f"Analyzing {len(args.tickers)} stock(s)...")
        print(f"{'='*80}\n")

        analyses = await analyzer.analyze_batch(args.tickers)

        if args.output == 'table':
            self._print_analysis_table(analyses)
        elif args.output == 'json':
            import json
            print(json.dumps([a.dict() for a in analyses], indent=2, default=str))
        elif args.output == 'csv':
            self._export_csv(analyses, 'analysis_results.csv')

    async def _handle_screen(
        self,
        analyzer: StockAnalyzer,
        provider_manager: ProviderManager,
        args
    ):
        """Handle screen command."""
        print(f"\n{'='*80}")
        print(f"Screening {args.universe.upper()} for top opportunities...")
        print(f"{'='*80}\n")

        # Fetch stock universe
        if args.universe == 'sp500':
            tickers = await provider_manager.fetch_sp500_tickers()
        else:
            # Placeholder for other universes
            tickers = await provider_manager.fetch_sp500_tickers()

        print(f"Analyzing {len(tickers)} stocks...\n")

        # Analyze all stocks
        analyses = await analyzer.analyze_batch(tickers, max_concurrent=50)

        # Filter by criteria
        filtered = [
            a for a in analyses
            if float(a.composite_score) >= args.min_score
        ]

        if args.signal:
            signal_type = SignalType(args.signal)
            filtered = [a for a in filtered if a.signal == signal_type]

        # Sort by score
        filtered.sort(key=lambda x: x.composite_score, reverse=True)

        # Take top N
        top_results = filtered[:args.top]

        print(f"\nFound {len(filtered)} stocks matching criteria")
        print(f"Showing top {len(top_results)} results:\n")

        if args.output == 'table':
            self._print_analysis_table(top_results)
        elif args.output == 'json':
            import json
            print(json.dumps([a.dict() for a in top_results], indent=2, default=str))
        elif args.output == 'csv':
            self._export_csv(top_results, 'screen_results.csv')

    def _print_analysis_table(self, analyses: List[Analysis]):
        """Print analyses in table format."""
        if not analyses:
            print("No results to display.")
            return

        # Header
        print(f"{'Ticker':<8} {'Score':<7} {'Signal':<12} {'Trend':<10} "
              f"{'Price':<10} {'RSI':<7} {'P/E':<7} {'ROE %':<8}")
        print("-" * 80)

        # Rows
        for analysis in analyses:
            ticker = analysis.ticker
            score = f"{float(analysis.composite_score):.1f}"
            signal = analysis.signal.value
            trend = analysis.trend.value

            price = f"${float(analysis.quote.price):.2f}" if analysis.quote else "N/A"

            rsi = (
                f"{float(analysis.technical_indicators.rsi):.1f}"
                if analysis.technical_indicators and analysis.technical_indicators.rsi
                else "N/A"
            )

            pe = (
                f"{float(analysis.fundamentals.pe_ratio):.1f}"
                if analysis.fundamentals and analysis.fundamentals.pe_ratio
                else "N/A"
            )

            roe = (
                f"{float(analysis.fundamentals.roe)*100:.1f}"
                if analysis.fundamentals and analysis.fundamentals.roe
                else "N/A"
            )

            print(f"{ticker:<8} {score:<7} {signal:<12} {trend:<10} "
                  f"{price:<10} {rsi:<7} {pe:<7} {roe:<8}")

        print("\n")

        # Show details for top result
        if analyses:
            top = analyses[0]
            print(f"Top Pick: {top.ticker}")
            print(f"  Score: {float(top.composite_score):.1f}/100")
            print(f"  Signal: {top.signal.value.upper()}")
            if top.key_strengths:
                print(f"  Strengths:")
                for strength in top.key_strengths[:3]:
                    print(f"    - {strength}")
            if top.key_risks:
                print(f"  Risks:")
                for risk in top.key_risks[:3]:
                    print(f"    - {risk}")
            print()

    def _export_csv(self, analyses: List[Analysis], filename: str):
        """Export analyses to CSV."""
        import pandas as pd

        data = []
        for a in analyses:
            data.append({
                'Ticker': a.ticker,
                'Score': float(a.composite_score),
                'Signal': a.signal.value,
                'Trend': a.trend.value,
                'Price': float(a.quote.price) if a.quote else None,
                'RSI': float(a.technical_indicators.rsi) if a.technical_indicators and a.technical_indicators.rsi else None,
                'PE': float(a.fundamentals.pe_ratio) if a.fundamentals and a.fundamentals.pe_ratio else None,
                'ROE': float(a.fundamentals.roe) if a.fundamentals and a.fundamentals.roe else None,
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


def main():
    """Main entry point."""
    cli = CLI()
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
