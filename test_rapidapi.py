"""Quick test of RapidAPI integration with correct endpoint."""
import asyncio
import sys
sys.path.insert(0, 'src')

from stock_analyzer.data.providers.yahoo import YahooFinanceProvider

async def test():
    provider = YahooFinanceProvider()
    await provider.initialize()

    print('Testing RapidAPI integration with correct endpoint...')
    print('=' * 60)

    test_tickers = ['AAPL', 'MSFT', 'GOOGL']

    for ticker in test_tickers:
        try:
            quote = await provider.fetch_quote(ticker)
            print(f'\n✓ {ticker}: ${quote.price} (Volume: {quote.volume:,})')
            if quote.change_percent:
                print(f'  Change: {quote.change_percent:.2f}%')
        except Exception as e:
            print(f'\n✗ {ticker}: ERROR - {e}')

    await provider.cleanup()
    print('\n' + '=' * 60)
    print('Test complete!')

if __name__ == '__main__':
    asyncio.run(test())
