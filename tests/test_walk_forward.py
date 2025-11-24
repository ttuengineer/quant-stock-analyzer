"""
Unit tests for walk-forward validation core functions.

Run with: pytest tests/test_walk_forward.py -v
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from walk_forward_validation import (
    bootstrap_pvalue,
    get_universe_at_date,
    normalize_ticker,
    set_ticker_aliases,
)


class TestBootstrapPvalue:
    """Tests for bootstrap_pvalue function."""

    def test_positive_returns_low_pvalue(self):
        """Consistently positive returns should have low p-value."""
        returns = np.array([0.02, 0.03, 0.01, 0.02, 0.01, 0.03, 0.02, 0.01])
        mean, p_value, ci_low, ci_high = bootstrap_pvalue(returns, n_iter=5000)

        assert mean > 0, "Mean should be positive"
        assert p_value < 0.05, "P-value should be significant"
        assert ci_low > 0, "Lower CI should be above 0"

    def test_zero_centered_returns_high_pvalue(self):
        """Returns centered around zero should have high p-value."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 50)  # Mean ~0
        mean, p_value, ci_low, ci_high = bootstrap_pvalue(returns, n_iter=5000)

        assert p_value > 0.3, f"P-value should be high for zero-centered returns, got {p_value}"

    def test_negative_returns_very_high_pvalue(self):
        """Negative returns should have p-value close to 1."""
        returns = np.array([-0.02, -0.01, -0.03, -0.02, -0.01])
        mean, p_value, ci_low, ci_high = bootstrap_pvalue(returns, n_iter=5000)

        assert mean < 0, "Mean should be negative"
        assert p_value > 0.95, "P-value should be very high"

    def test_confidence_interval_contains_mean(self):
        """95% CI should contain the sample mean."""
        np.random.seed(123)
        returns = np.random.normal(0.01, 0.02, 30)
        mean, p_value, ci_low, ci_high = bootstrap_pvalue(returns, n_iter=5000)

        # Mean should be within CI (this is almost always true)
        assert ci_low <= mean <= ci_high, "Mean should be within CI"


class TestNormalizeTicker:
    """Tests for ticker normalization."""

    def test_dot_to_dash_conversion(self):
        """BRK.B should match BRK-B."""
        variants = normalize_ticker("BRK.B")
        assert "BRK-B" in variants
        assert "BRK.B" in variants

    def test_dash_to_dot_conversion(self):
        """BRK-B should match BRK.B."""
        variants = normalize_ticker("BRK-B")
        assert "BRK.B" in variants
        assert "BRK-B" in variants

    def test_simple_ticker_unchanged(self):
        """Simple tickers like AAPL should include themselves."""
        variants = normalize_ticker("AAPL")
        assert "AAPL" in variants

    def test_alias_lookup(self):
        """Tickers with aliases should include all variants."""
        # Set up aliases
        set_ticker_aliases({
            "META": ["FB"],
            "FB": ["META"]
        })

        variants = normalize_ticker("META")
        assert "META" in variants
        assert "FB" in variants

        variants = normalize_ticker("FB")
        assert "FB" in variants
        assert "META" in variants


class TestGetUniverseAtDate:
    """Tests for universe lookup."""

    def test_exact_date_match(self):
        """Should return exact date's universe if available."""
        universe = {
            "2023-01-01": ["AAPL", "MSFT", "GOOGL"],
            "2023-06-01": ["AAPL", "MSFT", "META"],
        }

        result = get_universe_at_date(universe, "2023-01-01")
        assert result == {"AAPL", "MSFT", "GOOGL"}

    def test_between_dates(self):
        """Should return most recent universe before target date."""
        universe = {
            "2023-01-01": ["AAPL", "MSFT", "GOOGL"],
            "2023-06-01": ["AAPL", "MSFT", "META"],
        }

        # March should use January's universe
        result = get_universe_at_date(universe, "2023-03-15")
        assert result == {"AAPL", "MSFT", "GOOGL"}

    def test_after_all_dates(self):
        """Should return most recent universe for future dates."""
        universe = {
            "2023-01-01": ["AAPL", "MSFT", "GOOGL"],
            "2023-06-01": ["AAPL", "MSFT", "META"],
        }

        result = get_universe_at_date(universe, "2024-01-01")
        assert result == {"AAPL", "MSFT", "META"}

    def test_before_all_dates(self):
        """Should return earliest universe for dates before all snapshots."""
        universe = {
            "2023-01-01": ["AAPL", "MSFT", "GOOGL"],
            "2023-06-01": ["AAPL", "MSFT", "META"],
        }

        result = get_universe_at_date(universe, "2020-01-01")
        assert result == {"AAPL", "MSFT", "GOOGL"}

    def test_none_universe(self):
        """Should return None for None universe."""
        result = get_universe_at_date(None, "2023-01-01")
        assert result is None


class TestPriceLookup:
    """Tests for binary search price lookup."""

    def create_mock_ticker_data(self, dates, prices):
        """Create mock ticker data structure."""
        return {
            'dates': np.array(pd.to_datetime(dates).values),
            'prices': np.array(prices)
        }

    def test_get_price_exact_date(self):
        """Should return exact price when date exists."""
        # Import the function - it's defined inside walk_forward_validation
        # We'll test the logic directly
        dates = ['2023-01-02', '2023-01-03', '2023-01-04']
        prices = [100.0, 101.0, 102.0]
        ticker_data = self.create_mock_ticker_data(dates, prices)

        # Must convert to np.datetime64 for searchsorted comparison
        target = np.datetime64(pd.Timestamp('2023-01-03'))
        idx = np.searchsorted(ticker_data['dates'], target)

        if idx < len(ticker_data['dates']) and ticker_data['dates'][idx] == target:
            price = ticker_data['prices'][idx]
        else:
            price = None

        assert price == 101.0

    def test_get_price_forward_lookup(self):
        """Should return next available price for missing date (forward only)."""
        dates = ['2023-01-02', '2023-01-05', '2023-01-10']
        prices = [100.0, 105.0, 110.0]
        ticker_data = self.create_mock_ticker_data(dates, prices)

        # Target date is Jan 3 (not in data)
        # Must convert to np.datetime64 for searchsorted comparison
        target = np.datetime64(pd.Timestamp('2023-01-03'))
        idx = np.searchsorted(ticker_data['dates'], target)

        # Should get next available (Jan 5)
        if idx < len(ticker_data['dates']):
            price = ticker_data['prices'][idx]
            actual_date = ticker_data['dates'][idx]
        else:
            price = None
            actual_date = None

        assert price == 105.0
        assert pd.Timestamp(actual_date) == pd.Timestamp('2023-01-05')

    def test_no_backward_drift(self):
        """Should NOT return earlier price (no backward lookup)."""
        dates = ['2023-01-02', '2023-01-10']
        prices = [100.0, 110.0]
        ticker_data = self.create_mock_ticker_data(dates, prices)

        # Target date is Jan 5 - forward lookup should get Jan 10
        # Must convert to np.datetime64 for searchsorted comparison
        target = np.datetime64(pd.Timestamp('2023-01-05'))
        idx = np.searchsorted(ticker_data['dates'], target)

        if idx < len(ticker_data['dates']):
            price = ticker_data['prices'][idx]
        else:
            price = None

        # Should be Jan 10 price, NOT Jan 2
        assert price == 110.0

    def test_delisting_returns_none(self):
        """Should return None if no price available after target date."""
        dates = ['2023-01-02', '2023-01-03']
        prices = [100.0, 101.0]
        ticker_data = self.create_mock_ticker_data(dates, prices)

        # Target date is after all data (stock delisted)
        # Must convert to np.datetime64 for searchsorted comparison
        target = np.datetime64(pd.Timestamp('2023-06-01'))
        idx = np.searchsorted(ticker_data['dates'], target)

        if idx < len(ticker_data['dates']):
            price = ticker_data['prices'][idx]
        else:
            price = None

        assert price is None, "Should return None for delisted stock"


class TestICStats:
    """Tests for IC computation."""

    def test_perfect_positive_ic(self):
        """Perfect positive correlation should have IC = 1."""
        from walk_forward_validation import compute_ic_stats

        # Need 10+ samples for compute_ic_stats to calculate
        predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actuals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        stats = compute_ic_stats(predictions, actuals)
        assert abs(stats['ic'] - 1.0) < 0.01, "Perfect correlation should have IC=1"

    def test_perfect_negative_ic(self):
        """Perfect negative correlation should have IC = -1."""
        from walk_forward_validation import compute_ic_stats

        # Need 10+ samples for compute_ic_stats to calculate
        predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actuals = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

        stats = compute_ic_stats(predictions, actuals)
        assert abs(stats['ic'] + 1.0) < 0.01, "Perfect negative correlation should have IC=-1"

    def test_random_ic_near_zero(self):
        """Random data should have IC near 0."""
        from walk_forward_validation import compute_ic_stats

        np.random.seed(42)
        predictions = np.random.randn(100)
        actuals = np.random.randn(100)

        stats = compute_ic_stats(predictions, actuals)
        assert abs(stats['ic']) < 0.3, "Random data should have IC near 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
