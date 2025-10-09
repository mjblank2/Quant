"""Validation utilities for the Quant project.

This package contains modules that provide cross-validation routines,
probabilistic and deflated Sharpe ratio metrics, and other statistical
tools to help assess the robustness of trading strategies. Import
submodules as needed.
"""

from .cv import PurgedKFoldEmbargo, CombinatorialPurgedCV
from .metrics import sharpe_ratio, annualized_sharpe, psr, dsr

__all__ = [
    "PurgedKFoldEmbargo",
    "CombinatorialPurgedCV",
    "sharpe_ratio",
    "annualized_sharpe",
    "psr",
    "dsr",
]