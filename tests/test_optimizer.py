import pytest
import pandas as pd
import numpy as np
# Import the function to test
from portfolio.optimizer import build_pca_risk_model

@pytest.fixture
def sample_returns():
    """Generates sample returns data."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252)
    assets = [f'Asset_{i}' for i in range(20)]
    returns = pd.DataFrame(np.random.randn(252, 20) / 100, index=dates, columns=assets)
    return returns

def test_pca_risk_model_structure(sample_returns):
    """Tests the structure and shapes of the risk model components."""
    num_factors = 5
    risk_model = build_pca_risk_model(sample_returns, num_factors=num_factors)

    assert 'B' in risk_model and 'F' in risk_model and 'S_diag' in risk_model
    assert risk_model['B'].shape == (sample_returns.shape[1], num_factors)
    assert risk_model['F'].shape == (num_factors, num_factors)
    assert len(risk_model['S_diag']) == sample_returns.shape[1]

def test_pca_risk_model_properties(sample_returns):
    """Tests the mathematical properties of the risk model."""
    risk_model = build_pca_risk_model(sample_returns, num_factors=5)

    # Check if F (Factor Covariance) is positive semi-definite using Cholesky decomposition.
    try:
        np.linalg.cholesky(risk_model['F'])
    except np.linalg.LinAlgError:
        pytest.fail("Factor covariance matrix (F) is not positive semi-definite.")

    # Check if Idiosyncratic Risks (S_diag) are non-negative
    assert np.all(risk_model['S_diag'] >= 0)
