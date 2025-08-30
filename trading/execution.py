from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from config import DEFAULT_EXECUTION_SLICES, EXECUTION_STYLE
from tca.execution import TransactionCostModel
from tca.market_impact import SquareRootLaw

log = logging.getLogger(__name__)


@dataclass
class ChildOrder:
    """Child order for execution scheduling"""

    parent_id: int
    symbol: str
    side: str  # 'buy' or 'sell'
    slice_idx: int
    qty: int
    scheduled_time: datetime
    style: str
    target_price: Optional[float] = None
    participation_rate: Optional[float] = None


class ExecutionScheduler:
    """Advanced execution scheduling with VWAP, TWAP, and Implementation Shortfall algorithms"""

    def __init__(self):
        self.cost_model = TransactionCostModel()
        self.impact_model = SquareRootLaw()

    def schedule_child_orders(
        self, trades_df: pd.DataFrame, style: str = "twap", slices: int = None
    ) -> List[ChildOrder]:
        """
        Schedule child orders for a list of parent trades

        Args:
            trades_df: DataFrame with parent trades
            style: Execution algorithm ('vwap', 'twap', 'is')
            slices: Number of child orders (default from config)

        Returns:
            List of child orders
        """
        if trades_df.empty:
            return []

        slices = slices or DEFAULT_EXECUTION_SLICES
        child_orders = []

        for _, trade in trades_df.iterrows():
            try:
                if style.lower() == "vwap":
                    children = self._schedule_vwap(trade, slices)
                elif style.lower() == "twap":
                    children = self._schedule_twap(trade, slices)
                elif style.lower() == "is":
                    children = self._schedule_implementation_shortfall(trade, slices)
                else:
                    # Default to TWAP
                    children = self._schedule_twap(trade, slices)

                child_orders.extend(children)

            except Exception as e:
                log.error(
                    f"Failed to schedule child orders for {trade.get('symbol', 'unknown')}: {e}"
                )

        return child_orders

    def _schedule_twap(self, trade: pd.Series, slices: int) -> List[ChildOrder]:
        """Time-Weighted Average Price execution"""
        children = []

        total_qty = int(trade.get("quantity", 0))
        if total_qty == 0:
            return children

        parent_id = trade.get("id", 0)
        symbol = trade.get("symbol", "")
        side = trade.get("side", "buy")

        # Market hours: 9:30 AM to 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)

        # Calculate execution times
        execution_start = datetime.combine(datetime.now().date(), market_open)
        execution_end = datetime.combine(datetime.now().date(), market_close)

        # If current time is after market open, start from current time
        now = datetime.now()
        if now.time() > market_open and now.time() < market_close:
            execution_start = now

        total_minutes = (execution_end - execution_start).total_seconds() / 60
        slice_interval = total_minutes / slices

        # Equal-sized slices
        slice_qty = total_qty // slices
        remainder = total_qty % slices

        for i in range(slices):
            qty = slice_qty + (1 if i < remainder else 0)
            scheduled_time = execution_start + timedelta(minutes=i * slice_interval)

            child = ChildOrder(
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                slice_idx=i + 1,
                qty=qty,
                scheduled_time=scheduled_time,
                style="twap",
                participation_rate=0.10,  # Conservative participation rate
            )
            children.append(child)

        return children

    def _schedule_vwap(self, trade: pd.Series, slices: int) -> List[ChildOrder]:
        """Volume-Weighted Average Price execution"""
        children = []

        total_qty = int(trade.get("quantity", 0))
        if total_qty == 0:
            return children

        # Get historical volume profile
        symbol = trade.get("symbol", "")
        volume_profile = self._get_intraday_volume_profile(symbol)

        if volume_profile.empty:
            # Fall back to TWAP if no volume data
            return self._schedule_twap(trade, slices)

        parent_id = trade.get("id", 0)
        side = trade.get("side", "buy")

        # Allocate quantities based on volume profile
        total_volume = volume_profile["volume"].sum()
        volume_fractions = volume_profile["volume"] / total_volume

        # Market hours execution times
        market_open = time(9, 30)
        execution_start = datetime.combine(datetime.now().date(), market_open)

        for i, (time_bucket, vol_frac) in enumerate(
            volume_fractions.head(slices).items()
        ):
            qty = int(total_qty * vol_frac)
            if qty == 0:
                continue

            # Schedule based on time bucket
            minutes_offset = i * (390 / slices)  # 390 minutes in trading day
            scheduled_time = execution_start + timedelta(minutes=minutes_offset)

            # Higher participation rate during high volume periods
            participation_rate = min(0.20, 0.05 + vol_frac * 0.15)

            child = ChildOrder(
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                slice_idx=i + 1,
                qty=qty,
                scheduled_time=scheduled_time,
                style="vwap",
                participation_rate=participation_rate,
            )
            children.append(child)

        # Handle any remaining quantity in final slice
        allocated_qty = sum(child.qty for child in children)
        if allocated_qty < total_qty and children:
            children[-1].qty += total_qty - allocated_qty

        return children

    def _schedule_implementation_shortfall(
        self, trade: pd.Series, slices: int
    ) -> List[ChildOrder]:
        """Implementation Shortfall algorithm - optimal trade-off between market impact and timing risk"""
        children = []

        total_qty = int(trade.get("quantity", 0))
        if total_qty == 0:
            return children

        symbol = trade.get("symbol", "")
        price = trade.get("price", 100.0)

        # Get market data for optimization
        market_data = self._get_market_data(symbol)
        adv = market_data.get("adv", 1000000)
        volatility = market_data.get("volatility", 0.20)

        # Optimize execution schedule
        optimal_schedule = self._optimize_is_schedule(
            total_qty, price, adv, volatility, slices
        )

        parent_id = trade.get("id", 0)
        side = trade.get("side", "buy")

        market_open = time(9, 30)
        execution_start = datetime.combine(datetime.now().date(), market_open)

        for i, (qty, time_offset, part_rate) in enumerate(optimal_schedule):
            if qty <= 0:
                continue

            scheduled_time = execution_start + timedelta(hours=time_offset)

            child = ChildOrder(
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                slice_idx=i + 1,
                qty=qty,
                scheduled_time=scheduled_time,
                style="is",
                participation_rate=part_rate,
            )
            children.append(child)

        return children

    def _get_intraday_volume_profile(self, symbol: str) -> pd.DataFrame:
        """Get historical intraday volume profile"""
        try:
            # Simple volume profile - could be enhanced with actual intraday data
            # For now, return typical U-shaped pattern
            times = [
                "09:30",
                "10:00",
                "10:30",
                "11:00",
                "11:30",
                "12:00",
                "12:30",
                "13:00",
                "13:30",
                "14:00",
                "14:30",
                "15:00",
                "15:30",
                "16:00",
            ]

            # U-shaped volume profile (high at open/close, low at midday)
            volume_multipliers = [
                1.5,
                1.2,
                1.0,
                0.8,
                0.7,
                0.6,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.2,
                1.4,
                1.6,
            ]

            profile = pd.DataFrame({"time": times, "volume": volume_multipliers})
            profile = profile.set_index("time")

            return profile

        except Exception as e:
            log.warning(f"Could not load volume profile for {symbol}: {e}")
            return pd.DataFrame()

    def _get_market_data(self, symbol: str) -> Dict:
        """Get market data for execution optimization"""
        try:
            from db import engine
            from sqlalchemy import text

            # Get recent market data
            stmt = text(
                """
                SELECT AVG(volume * close) as adv,
                       STDDEV(close/LAG(close) OVER (ORDER BY ts) - 1) as vol
                FROM daily_bars
                WHERE symbol = :symbol
                AND ts >= CURRENT_DATE - INTERVAL '20 days'
            """
            )

            result = engine.execute(stmt, symbol=symbol).fetchone()

            if result:
                return {
                    "adv": result[0] or 1000000,
                    "volatility": (result[1] or 0.20) * np.sqrt(252),
                }

        except Exception as e:
            log.warning(f"Could not load market data for {symbol}: {e}")

        # Default values
        return {"adv": 1000000, "volatility": 0.20}

    def _optimize_is_schedule(
        self, total_qty: int, price: float, adv: float, volatility: float, slices: int
    ) -> List[Tuple[int, float, float]]:
        """Optimize Implementation Shortfall execution schedule"""

        # Simplified IS optimization
        # In practice, this would solve a dynamic programming problem

        trading_hours = 6.5  # 9:30 AM to 4:00 PM
        time_step = trading_hours / slices

        schedule = []
        remaining_qty = total_qty

        for i in range(slices):
            # Time-decaying execution: front-load more aggressive execution
            # to reduce timing risk, but balance against market impact

            time_factor = 1.0 - (i / slices)  # Decreasing over time
            urgency = 0.3 + 0.7 * time_factor  # Higher urgency early

            # Optimal slice size based on trade-off
            if i == slices - 1:
                slice_qty = remaining_qty  # Execute remaining
            else:
                # Fraction decreases over time (front-loaded)
                fraction = 2.0 / slices * (1.0 - i / (2 * slices))
                slice_qty = int(total_qty * fraction)
                slice_qty = min(slice_qty, remaining_qty)

            time_offset = i * time_step

            # Participation rate based on urgency and market impact
            order_rate = slice_qty / adv
            if order_rate < 0.01:
                part_rate = min(0.30, urgency * 0.40)
            elif order_rate < 0.05:
                part_rate = min(0.20, urgency * 0.25)
            else:
                part_rate = min(0.15, urgency * 0.20)

            schedule.append((slice_qty, time_offset, part_rate))
            remaining_qty -= slice_qty

            if remaining_qty <= 0:
                break

        return schedule


# Global function for backward compatibility
def schedule_child_orders(
    trades_df: pd.DataFrame, style: str = None, slices: int = None
) -> pd.DataFrame:
    """
    Schedule child orders for execution

    Args:
        trades_df: DataFrame of parent trades
        style: Execution style ('vwap', 'twap', 'is')
        slices: Number of child orders

    Returns:
        DataFrame of child orders
    """
    style = style or EXECUTION_STYLE
    scheduler = ExecutionScheduler()

    child_orders = scheduler.schedule_child_orders(trades_df, style, slices)

    if not child_orders:
        return pd.DataFrame()

    # Convert to DataFrame
    records = []
    for child in child_orders:
        records.append(
            {
                "parent_id": child.parent_id,
                "symbol": child.symbol,
                "side": child.side,
                "slice_idx": child.slice_idx,
                "qty": child.qty,
                "scheduled_time": child.scheduled_time,
                "style": child.style,
                "target_price": child.target_price,
                "participation_rate": child.participation_rate,
            }
        )

    return pd.DataFrame(records)
