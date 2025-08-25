from __future__ import annotations
import numpy as np, pandas as pd
try:
    import cvxpy as cp
except Exception:
    cp = None

def solve_qp(expected: pd.Series, C: np.ndarray, gross: float, w_cap: float) -> pd.Series | None:
    if cp is None:
        return None
    n = len(expected)
    w = cp.Variable(n)
    mu = expected.values
    lam = 1.0  # will be scaled outside
    obj = cp.Maximize(mu @ w - lam * cp.quad_form(w, C))
    cons = [
        cp.norm1(w) <= gross,
        w >= -w_cap,
        w <= w_cap
    ]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, warm_start=True, max_iters=10000)
    except Exception:
        return None
    if w.value is None:
        return None
    return pd.Series(np.array(w.value).flatten(), index=expected.index)
