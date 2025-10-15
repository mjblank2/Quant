"""
Broker wrapper for the Quant system using Interactive Brokers only.

This module previously contained Alpaca and FIX integrations. Alpaca support has been removed.
Use sync_trades_to_broker(...) to submit generated trades via IB.
"""

from trading.broker_ib import sync_trades_to_ib as sync_trades_to_broker

__all__ = ["sync_trades_to_broker"]
