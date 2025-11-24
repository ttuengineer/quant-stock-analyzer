"""
Fetch Historical S&P 500 Membership to Address Survivorship Bias.

Based on ChatGPT's recommendation:
- Scrape Wikipedia's S&P 500 changes table
- Build historical membership by working backwards
- This removes 70-80% of survivorship bias for FREE

Survivorship bias impact: typically 3-7% per year in stock selection models.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def fetch_sp500_changes():
    """
    Fetch S&P 500 historical changes from Wikipedia.
    Returns DataFrame with: date, added, removed, reason
    """
    print("=" * 60)
    print("FETCHING HISTORICAL S&P 500 CHANGES")
    print("=" * 60)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"\nFetching from: {url}")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all wikitables - second one has changes
    tables = soup.find_all('table', {'class': 'wikitable'})

    if len(tables) < 2:
        print("ERROR: Could not find S&P 500 changes table")
        return None

    changes_table = tables[1]
    changes = []
    rows = changes_table.find_all('tr')[1:]

    print(f"Found {len(rows)} change records")

    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 4:
            try:
                date_str = cols[0].get_text(strip=True)
                added = cols[1].get_text(strip=True)
                removed = cols[3].get_text(strip=True)
                reason = cols[4].get_text(strip=True) if len(cols) > 4 else ""

                try:
                    date = pd.to_datetime(date_str)
                except:
                    continue

                # Clean ticker symbols
                added = added.split('[')[0].strip() if added else ""
                removed = removed.split('[')[0].strip() if removed else ""

                if added or removed:
                    changes.append({
                        'date': date,
                        'added': added,
                        'removed': removed,
                        'reason': reason
                    })
            except:
                continue

    df = pd.DataFrame(changes)
    df = df.sort_values('date').reset_index(drop=True)

    print(f"\nParsed {len(df)} valid changes")
    if len(df) > 0:
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def fetch_current_sp500():
    """Fetch current S&P 500 constituents with sectors."""
    print("\nFetching current S&P 500 members...")

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table', {'class': 'wikitable'})

    tickers = []
    sectors = {}

    rows = table.find_all('tr')[1:]
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 4:
            ticker = cols[0].get_text(strip=True)
            sector = cols[3].get_text(strip=True) if len(cols) > 3 else ""

            ticker = ticker.replace('.', '-')  # BRK.B -> BRK-B
            tickers.append(ticker)
            sectors[ticker] = sector

    print(f"Found {len(tickers)} current S&P 500 members")
    return tickers, sectors


def build_historical_universe(changes_df, current_members, start_date='2015-01-01'):
    """
    Build historical universe by working backwards from current members.

    This is ChatGPT's recommended approach:
    - Start with current S&P 500
    - Work backwards through each change
    - Reverse additions (remove from earlier universe)
    - Reverse removals (add back to earlier universe)
    """
    print("\n" + "=" * 60)
    print("BUILDING HISTORICAL UNIVERSE")
    print("=" * 60)

    start = pd.to_datetime(start_date)
    today = pd.to_datetime(datetime.now().date())

    universe = set(current_members)
    historical_universe = {}

    # Current state
    historical_universe[today.strftime('%Y-%m-%d')] = list(universe)

    # Work backwards through changes
    changes = changes_df[changes_df['date'] >= start].sort_values('date', ascending=False)

    for _, change in changes.iterrows():
        date = change['date']
        added = change['added']
        removed = change['removed']

        # Reverse the change (going backwards in time)
        if added and added in universe:
            universe.remove(added)
        if removed:
            universe.add(removed)

        historical_universe[date.strftime('%Y-%m-%d')] = list(universe.copy())

    print(f"Built universe snapshots for {len(historical_universe)} dates")

    # Show sample sizes
    print("\nHistorical universe sizes:")
    for year in [2015, 2018, 2020, 2022, 2024]:
        target = f"{year}-01-01"
        # Find nearest date <= target
        dates = sorted([d for d in historical_universe.keys() if d <= target], reverse=True)
        if dates:
            nearest = dates[0]
            print(f"  {year}: {len(historical_universe[nearest])} stocks (as of {nearest})")

    return historical_universe


def get_universe_at_date(historical_universe, target_date):
    """Get the S&P 500 universe at a specific date."""
    target = pd.to_datetime(target_date).strftime('%Y-%m-%d')

    # Find the most recent universe snapshot <= target date
    dates = sorted([d for d in historical_universe.keys() if d <= target], reverse=True)

    if dates:
        return set(historical_universe[dates[0]])

    # Fallback to earliest available
    earliest = min(historical_universe.keys())
    return set(historical_universe[earliest])


def build_ticker_aliases(changes_df):
    """
    Build ticker alias map from changes data.

    When a stock changes ticker (e.g., FB -> META), the changes table
    shows them on the same row. We capture these relationships.
    """
    aliases = {}

    # Known ticker changes that appear as same-day add/remove
    # These are stocks that changed ticker symbols, not actual replacements
    ticker_change_keywords = [
        'ticker change', 'symbol change', 'renamed', 'rebranded',
        'changed ticker', 'ticker symbol change'
    ]

    for _, row in changes_df.iterrows():
        added = row['added']
        removed = row['removed']
        reason = str(row.get('reason', '')).lower()

        # If both added and removed on same date, might be ticker change
        if added and removed:
            # Check if reason indicates ticker change
            is_ticker_change = any(kw in reason for kw in ticker_change_keywords)

            # Also check for common patterns (same company, different ticker)
            # e.g., FB removed, META added = same company
            if is_ticker_change:
                # Add bidirectional mapping
                if added not in aliases:
                    aliases[added] = set()
                if removed not in aliases:
                    aliases[removed] = set()
                aliases[added].add(removed)
                aliases[removed].add(added)

    # Add common format variations (BRK.B <-> BRK-B)
    all_tickers = set()
    for key in list(aliases.keys()):
        all_tickers.add(key)
        all_tickers.update(aliases[key])

    # Convert sets to lists for JSON serialization
    result = {k: list(v) for k, v in aliases.items()}

    # Add format variations for tickers with . or -
    format_aliases = {}
    for ticker in all_tickers:
        if '.' in ticker or '-' in ticker:
            alt1 = ticker.replace('.', '-')
            alt2 = ticker.replace('-', '.')
            if ticker not in format_aliases:
                format_aliases[ticker] = []
            if alt1 != ticker:
                format_aliases[ticker].append(alt1)
            if alt2 != ticker:
                format_aliases[ticker].append(alt2)

    # Merge format aliases
    for ticker, alts in format_aliases.items():
        if ticker not in result:
            result[ticker] = alts
        else:
            result[ticker] = list(set(result[ticker] + alts))

    return result


def save_universe(historical_universe, sectors, changes_df):
    """Save historical universe and metadata."""
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Count survivorship bias candidates
    all_removed = set()
    for _, change in changes_df.iterrows():
        if change['removed']:
            all_removed.add(change['removed'])

    current = set(historical_universe[max(historical_universe.keys())])
    removed_permanently = all_removed - current

    # Build ticker aliases from changes data
    ticker_aliases = build_ticker_aliases(changes_df)
    print(f"\nBuilt {len(ticker_aliases)} ticker aliases from changes data")

    output = {
        'generated': datetime.now().isoformat(),
        'source': 'Wikipedia S&P 500 changes table',
        'note': 'Built by working backwards from current members',
        'survivorship_bias_stocks': list(removed_permanently),
        'ticker_aliases': ticker_aliases,  # Dynamic alias map!
        'sectors': sectors,
        'universe_by_date': historical_universe
    }

    output_path = output_dir / "historical_sp500_universe.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to: {output_path}")
    return output_path


def main():
    """Build and save historical S&P 500 universe."""
    print("\n" + "=" * 70)
    print("S&P 500 HISTORICAL MEMBERSHIP BUILDER")
    print("Fixes survivorship bias (ChatGPT recommendation)")
    print("=" * 70)

    # Step 1: Fetch current members
    current_members, sectors = fetch_current_sp500()

    # Step 2: Fetch historical changes
    changes_df = fetch_sp500_changes()

    if changes_df is None or len(changes_df) == 0:
        print("ERROR: Could not fetch historical changes")
        return None

    # Step 3: Build historical universe
    historical_universe = build_historical_universe(
        changes_df,
        current_members,
        start_date='2015-01-01'
    )

    # Step 4: Save
    output_path = save_universe(historical_universe, sectors, changes_df)

    # Summary
    print("\n" + "=" * 70)
    print("SURVIVORSHIP BIAS ANALYSIS")
    print("=" * 70)

    all_removed = set()
    for _, change in changes_df.iterrows():
        if change['removed']:
            all_removed.add(change['removed'])

    current = set(current_members)
    removed_permanently = all_removed - current

    print(f"\nStocks removed from S&P 500 since 2015: {len(all_removed)}")
    print(f"Stocks removed and NOT re-added: {len(removed_permanently)}")
    print(f"\nThese {len(removed_permanently)} stocks create survivorship bias!")
    print(f"(Typically adds 3-7% per year to backtested returns)")

    if removed_permanently:
        print(f"\nExamples of removed stocks:")
        for ticker in list(removed_permanently)[:15]:
            print(f"  - {ticker}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Update walk_forward_validation.py to filter by historical universe")
    print("2. Re-run: python scripts/walk_forward_validation.py")
    print("3. Compare results (expect ~3-7% lower annual returns)")

    return historical_universe


if __name__ == "__main__":
    main()
