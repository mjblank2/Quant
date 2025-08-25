from __future__ import annotations
import numpy as np
import pandas as pd

def optimize_qp(pred_df: pd.DataFrame,
                long_budget: float,
                max_per_name: float,
                factor_cols = ('size_ln','mom_21','turnover_21','beta_63'),
                corr_penalty: float = 0.05):
    """Simple convex optimizer for long-only weights with crowding penalty.
    If cvxpy is unavailable, returns equal-weight solution within caps.
    """
    try:
        import cvxpy as cp
    except Exception:
        # Fallback: equal-weighted top names within cap
        n = len(pred_df)
        if n == 0: return pd.Series(dtype=float)
        w = np.full(n, long_budget / n)
        w = np.minimum(w, max_per_name)
        w *= (long_budget / max(w.sum(), 1e-12))
        return pd.Series(w, index=pred_df['symbol'].values)

    scores = pred_df['y_pred'].values
    X = pred_df[[c for c in factor_cols if c in pred_df.columns]].fillna(0.0).values
    n = len(scores)
    w = cp.Variable(n)
    # Similarity penalty via factor covariance proxy
    if X.shape[1] > 0:
        Q = X @ X.T
        Q = Q / (np.linalg.norm(Q) + 1e-8)
    else:
        Q = np.eye(n)
    # Objective: maximize scores'w - corr_penalty * w'Qw
    obj = cp.Maximize(scores @ w - corr_penalty * cp.quad_form(w, Q))
    constraints = [
        w >= 0,
        w <= max_per_name,
        cp.sum(w) == long_budget
    ]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
    except Exception:
        prob.solve(warm_start=True, verbose=False)
    if w.value is None:
        # fallback
        w0 = np.full(n, long_budget / n)
        w0 = np.minimum(w0, max_per_name)
        w0 *= (long_budget / max(w0.sum(), 1e-12))
        return pd.Series(w0, index=pred_df['symbol'].values)
    return pd.Series(np.asarray(w.value).ravel(), index=pred_df['symbol'].values)