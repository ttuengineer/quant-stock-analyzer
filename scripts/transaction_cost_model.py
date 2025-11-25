"""
Sophisticated Transaction Cost Modeling.

Models realistic trading costs including:
- Bid-ask spread
- Market impact (price moves when you trade)
- Commission fees
- Slippage
- Volume-based analysis

This gives more realistic backtest results than flat cost assumptions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class TransactionCostModel:
    """
    Models transaction costs for realistic backtesting.
    """

    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% commission
        min_commission: float = 1.0,  # $1 minimum
        market_impact_coef: float = 0.0001  # Market impact coefficient
    ):
        """
        Initialize cost model.

        Args:
            commission_rate: Commission as fraction of trade value
            min_commission: Minimum commission per trade
            market_impact_coef: Market impact coefficient
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.market_impact_coef = market_impact_coef

    def estimate_spread_cost(
        self,
        ticker: str,
        price: float,
        volume: Optional[float] = None
    ) -> float:
        """
        Estimate bid-ask spread cost.

        Spread varies by liquidity:
        - Large cap (S&P 500): 0.01-0.05%
        - Mid cap: 0.05-0.15%
        - Small cap: 0.15-0.50%

        Args:
            ticker: Stock ticker
            price: Current price
            volume: Average daily volume

        Returns:
            Spread cost as fraction
        """
        # Estimate spread based on price (rough heuristic)
        if price > 100:
            # Large cap, tight spread
            spread_bps = 2  # 2 basis points
        elif price > 50:
            # Mid cap
            spread_bps = 5
        elif price > 20:
            spread_bps = 10
        else:
            # Small cap, wide spread
            spread_bps = 20

        spread_fraction = spread_bps / 10000  # Convert basis points to fraction

        return spread_fraction

    def estimate_market_impact(
        self,
        trade_value: float,
        avg_daily_volume: float = 1000000,
        volatility: float = 0.02
    ) -> float:
        """
        Estimate market impact cost.

        When you trade, you move the price. Larger trades have bigger impact.

        Square-root model: impact proportional to sqrt(trade_size / daily_volume) * volatility

        Args:
            trade_value: Dollar value of trade
            avg_daily_volume: Average daily dollar volume
            volatility: Stock volatility

        Returns:
            Market impact as fraction of trade value
        """
        if avg_daily_volume <= 0:
            return 0.0

        participation_rate = trade_value / avg_daily_volume

        # Square-root impact model (common in academic literature)
        impact = self.market_impact_coef * np.sqrt(participation_rate) * volatility

        # Cap at reasonable level
        impact = min(impact, 0.05)  # Max 5% impact

        return impact

    def estimate_total_cost(
        self,
        trade_value: float,
        price: float,
        ticker: str = "",
        avg_daily_volume: float = 1000000,
        volatility: float = 0.02
    ) -> Dict[str, float]:
        """
        Estimate total transaction cost.

        Args:
            trade_value: Dollar value of trade
            price: Stock price
            ticker: Stock ticker
            avg_daily_volume: Average daily volume
            volatility: Stock volatility

        Returns:
            Dictionary with cost breakdown
        """
        # Commission
        commission = max(trade_value * self.commission_rate, self.min_commission)
        commission_fraction = commission / trade_value if trade_value > 0 else 0

        # Spread
        spread = self.estimate_spread_cost(ticker, price)

        # Market impact
        impact = self.estimate_market_impact(trade_value, avg_daily_volume, volatility)

        # Total cost
        total_fraction = commission_fraction + spread + impact

        return {
            'commission': commission_fraction,
            'spread': spread,
            'market_impact': impact,
            'total': total_fraction,
            'total_bps': total_fraction * 10000  # In basis points
        }

    def apply_costs_to_returns(
        self,
        returns: pd.Series,
        turnover_rate: float
    ) -> pd.Series:
        """
        Apply transaction costs to return series.

        Args:
            returns: Series of portfolio returns
            turnover_rate: Monthly turnover rate (0-1)

        Returns:
            Returns after costs
        """
        # Average cost per trade
        avg_cost = 0.0015  # 15 basis points typical

        # Cost per period = turnover * avg_cost
        cost_per_period = turnover_rate * avg_cost

        # Apply costs
        returns_after_costs = returns - cost_per_period

        return returns_after_costs
