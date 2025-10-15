import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import cvxpy as cp


def build_pca_risk_model(returns: pd.DataFrame, num_factors: int = 15) -> dict:
    """
    Builds a statistical multi-factor risk model using PCA.
    Returns a dictionary with factor exposures (B), factor covariance (F), and specific variances (S_diag).

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns with dates as index and symbols as columns.
    num_factors : int, optional
        Number of principal components (factors) to extract.
    """
    # Handle missing data and demean returns
    returns_filled = returns.fillna(0)
    demeaned_returns = returns_filled - returns_filled.mean()

    pca = PCA(n_components=num_factors)
    pca.fit(demeaned_returns)

    # Factor exposures (B)
    factor_exposures = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f'Factor_{i+1}' for i in range(num_factors)]
    )

    # Factor returns matrix
    factor_returns = pca.transform(demeaned_returns)

    # Factor covariance matrix (F)
    factor_covariance = np.cov(factor_returns, rowvar=False, ddof=1)

    # Idiosyncratic risk (specific variances)
    reconstructed_returns = factor_returns @ factor_exposures.T
    residuals = demeaned_returns.values - reconstructed_returns
    idiosyncratic_risk_diag = np.var(residuals, axis=0, ddof=1)

    risk_model = {
        'B': factor_exposures,
        'F': factor_covariance,
        'S_diag': idiosyncratic_risk_diag
    }
    return risk_model


def optimize_portfolio_with_tca(alpha: pd.Series, risk_model: dict, current_weights: pd.Series,
                                tca_rate: float = 0.0005, risk_aversion: float = 1.0,
                                max_leverage: float = 1.0) -> np.ndarray:
    """
    Performs mean-variance optimization with transaction cost awareness using a factor risk model.

    Parameters
    ----------
    alpha : pd.Series
        Expected returns or alpha signals indexed by symbol.
    risk_model : dict
        Risk model dictionary containing 'B', 'F', and 'S_diag' from build_pca_risk_model().
    current_weights : pd.Series
        Current portfolio weights indexed by symbol.
    tca_rate : float, optional
        Linear transaction cost rate per dollar traded.
    risk_aversion : float, optional
        Risk aversion parameter that controls trade-off between return and risk.
    max_leverage : float, optional
        Maximum gross leverage allowed.

    Returns
    -------
    np.ndarray
        Optimized weights vector for the assets in alpha.index.
    """
    n_assets = len(alpha)
    # Align indices
    if not alpha.index.equals(risk_model['B'].index):
        alpha = alpha.loc[risk_model['B'].index]
    current_weights = current_weights.reindex(alpha.index).fillna(0.0)

    weights = cp.Variable(n_assets)

    # Portfolio return
    portfolio_return = alpha.values.T @ weights

    # Portfolio risk using factor model
    B = risk_model['B'].values
    F = risk_model['F']
    S_diag = risk_model['S_diag']

    factor_exposures = B.T @ weights
    factor_risk = cp.quad_form(factor_exposures, F)
    idiosyncratic_risk = cp.sum(cp.multiply(S_diag, cp.square(weights)))
    portfolio_risk = factor_risk + idiosyncratic_risk

    # Transaction costs
    transaction_costs = cp.sum(cp.abs(weights - current_weights.values) * tca_rate)

    # Objective: maximize return minus risk and transaction costs
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk - transaction_costs)

    # Constraints: Dollar neutral and leverage limit
    constraints = [
        cp.sum(weights) == 0,
        cp.norm(weights, 1) <= max_leverage
    ]

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS)
    except cp.SolverError:
        try:
            problem.solve(solver=cp.SCS)
        except cp.SolverError:
            return current_weights.values

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or weights.value is None:
        return current_weights.values

    return weights.value
