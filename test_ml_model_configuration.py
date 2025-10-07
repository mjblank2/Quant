"""Tests safeguarding core model configuration defaults.

These tests ensure that critical guardrails, such as the validated loss for the
HistGradientBoostingRegressor models, remain intact as the training pipeline
continues to evolve.
"""

import importlib
import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import models.ml as ml


def test_hist_gradient_boosting_loss_is_valid():
    """The configured HistGradientBoostingRegressor loss must match the module constant."""
    specs = ml._model_specs()
    assert "hgb" in specs, "HistGradientBoostingRegressor spec should be available"
    estimator = specs["hgb"].named_steps["model"]
    assert estimator.get_params()["loss"] == ml.HGB_LOSS
    assert ml.HGB_LOSS in ml._HGB_VALID_LOSSES


def test_invalid_env_loss_falls_back(monkeypatch: pytest.MonkeyPatch):
    """An invalid HGB_LOSS environment value should log and fall back to squared_error."""
    monkeypatch.setenv("HGB_LOSS", "lsquared_error")
    reloaded = importlib.reload(ml)
    try:
        assert reloaded.HGB_LOSS == "squared_error"
        specs = reloaded._model_specs()
        estimator = specs["hgb"].named_steps["model"]
        assert estimator.get_params()["loss"] == "squared_error"
    finally:
        monkeypatch.delenv("HGB_LOSS", raising=False)
        importlib.reload(reloaded)
