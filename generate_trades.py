"""Convenience wrapper to generate today's trades.

This script imports the core generate_today_trades function from the
trading module and executes it when run as a standalone program.
"""
from trading.generate_trades import generate_today_trades


if __name__ == "__main__":
    generate_today_trades()
