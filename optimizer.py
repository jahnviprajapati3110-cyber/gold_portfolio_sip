"""
optimizer.py
------------
Constrained mean-variance optimization: minimum-variance portfolio,
maximum-Sharpe (tangency) portfolio, and the efficient frontier, all under
long-only weights that sum to 1 and a configurable minimum-allocation floor
per asset (Joshipura & Joshipura's constraint). Also the 3-year rolling
re-estimation used to show regime drift in the optimal gold weight.

This is a genuinely constrained quadratic optimization problem (inequality
constraints on each weight), which is why it needs scipy rather than plain
formulas -- see the note in the workbook this project replaces.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config


# ---------------------------------------------------------------------------
# Mean / covariance
# ---------------------------------------------------------------------------

def annualized_mean_cov(returns_df: pd.DataFrame, asset_cols: list) -> tuple:
    """Arithmetic monthly mean x 12, and monthly covariance x 12.

    Note: portfolio theory conventionally uses the ARITHMETIC mean (not the
    geometric/CAGR mean) as the expected-return input to a mean-variance
    optimizer, because portfolio variance is also derived under the
    arithmetic/linear framework. Use a separate geometric-return function
    (see cagr) if you want a "what actually happened" performance number
    for reporting instead.
    """
    R = returns_df[asset_cols]
    mean_annual = R.mean().values * 12
    cov_annual = R.cov().values * 12
    return mean_annual, cov_annual


def cagr(returns_df: pd.DataFrame, asset_cols: list) -> np.ndarray:
    """Geometric/compounded annualized return -- use this for reporting
    'what did this asset actually deliver per year', not for the optimizer.
    """
    R = returns_df[asset_cols]
    n = len(R)
    growth = (1 + R).prod()
    return (growth ** (12 / n) - 1).values


def average_risk_free(returns_df: pd.DataFrame, rf_col: str = "risk_free") -> tuple:
    """Average monthly risk-free rate, annualized, plus a coverage string
    so you always know how much of the window actually had T-Bill data.
    """
    rf = returns_df[rf_col].dropna()
    if len(rf) == 0:
        return None, "0 months (no T-Bill data)"
    return rf.mean() * 12, f"{len(rf)}/{len(returns_df)} months"


# ---------------------------------------------------------------------------
# Constrained optimization
# ---------------------------------------------------------------------------

def _bounds(n_assets: int, floor: float):
    # Each weight in [floor, 1 - (n_assets-1)*floor] -- the upper bound is
    # what's left over once every OTHER asset is held at its floor.
    return [(floor, 1 - (n_assets - 1) * floor)] * n_assets


def _sum_to_one_constraint():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1}


def min_variance(cov: np.ndarray, floor: float = config.DEFAULT_FLOOR) -> np.ndarray:
    """Weights that minimize portfolio variance, long-only, floor-constrained."""
    n = len(cov)
    w0 = np.array([1 / n] * n)
    res = minimize(lambda w, c: w @ c @ w, w0, args=(cov,), method="SLSQP",
                    bounds=_bounds(n, floor), constraints=[_sum_to_one_constraint()])
    if not res.success:
        raise RuntimeError(f"min_variance did not converge: {res.message}")
    return res.x


def min_variance_for_target(cov: np.ndarray, mean: np.ndarray, target: float,
                             floor: float = config.DEFAULT_FLOOR):
    """Weights that minimize variance subject to hitting a target return.
    Used to trace the efficient frontier. Returns None if infeasible at
    this target return given the floor constraint.
    """
    n = len(cov)
    w0 = np.array([1 / n] * n)
    cons = [_sum_to_one_constraint(),
            {"type": "eq", "fun": lambda w: w @ mean - target}]
    res = minimize(lambda w, c: w @ c @ w, w0, args=(cov,), method="SLSQP",
                    bounds=_bounds(n, floor), constraints=cons)
    return res.x if res.success else None


def tangency(mean: np.ndarray, cov: np.ndarray, risk_free: float,
             floor: float = config.DEFAULT_FLOOR) -> np.ndarray:
    """Weights that maximize the Sharpe ratio (return - rf) / std."""
    n = len(cov)
    w0 = np.array([1 / n] * n)

    def neg_sharpe(w):
        ret = w @ mean
        vol = np.sqrt(w @ cov @ w)
        return -(ret - risk_free) / vol

    res = minimize(neg_sharpe, w0, method="SLSQP",
                    bounds=_bounds(n, floor), constraints=[_sum_to_one_constraint()])
    if not res.success:
        raise RuntimeError(f"tangency did not converge: {res.message}")
    return res.x


def portfolio_stats(weights: np.ndarray, mean: np.ndarray, cov: np.ndarray,
                     risk_free: float = None) -> dict:
    ret = float(weights @ mean)
    std = float(np.sqrt(weights @ cov @ weights))
    out = {"weights": weights, "return": ret, "std": std}
    if risk_free is not None:
        out["sharpe"] = (ret - risk_free) / std
    return out


def efficient_frontier(mean: np.ndarray, cov: np.ndarray, floor: float = config.DEFAULT_FLOOR,
                        n_points: int = config.DEFAULT_FRONTIER_POINTS) -> list:
    """Trace the frontier by looping over achievable target returns (the
    feasible range given the floor constraint) and minimizing variance at
    each one. Returns a list of dicts: target_return, std, weights.
    """
    n = len(mean)
    # Feasible return range: the vertices of the floor-constrained weight
    # simplex are "one asset at its ceiling, the rest at the floor".
    vertex_returns = []
    for i in range(n):
        w = np.array([floor] * n)
        w[i] = 1 - (n - 1) * floor
        vertex_returns.append(w @ mean)
    lo, hi = min(vertex_returns), max(vertex_returns)

    frontier = []
    for target in np.linspace(lo, hi, n_points):
        w = min_variance_for_target(cov, mean, target, floor)
        if w is not None:
            frontier.append({
                "target_return": float(target),
                "std": float(np.sqrt(w @ cov @ w)),
                "weights": w,
            })
    return frontier


def equal_weight_portfolio(mean: np.ndarray, cov: np.ndarray, risk_free: float = None) -> dict:
    n = len(mean)
    w = np.array([1 / n] * n)
    return portfolio_stats(w, mean, cov, risk_free)


# ---------------------------------------------------------------------------
# Rolling window optimization (Phase 4)
# ---------------------------------------------------------------------------

def rolling_tangency_weights(returns_df: pd.DataFrame, asset_cols: list,
                              window_months: int = config.DEFAULT_ROLLING_WINDOW_MONTHS,
                              floor: float = config.DEFAULT_FLOOR,
                              rf_col: str = "risk_free") -> pd.DataFrame:
    """
    Step forward one month at a time. At each step, use the trailing
    `window_months` of data to re-estimate the mean vector, covariance
    matrix, and average risk-free rate, then solve for the tangency
    portfolio. Returns a DataFrame indexed by date with one column per
    asset (the optimal weight that month) plus 'return', 'std', 'sharpe'.

    If the trailing window has no risk-free data at all, that month is
    skipped (recorded as NaN) rather than silently using a wrong number --
    decide explicitly if you'd rather fall back to a default here.
    """
    dates = returns_df.index
    records = []
    for i in range(window_months, len(dates) + 1):
        window = returns_df.iloc[i - window_months:i]
        mean, cov = annualized_mean_cov(window, asset_cols)
        rf, rf_coverage = average_risk_free(window, rf_col)
        as_of = dates[i - 1]

        if rf is None:
            records.append({"date": as_of, **{a: np.nan for a in asset_cols},
                             "return": np.nan, "std": np.nan, "sharpe": np.nan,
                             "rf_coverage": rf_coverage})
            continue

        w = tangency(mean, cov, rf, floor)
        stats = portfolio_stats(w, mean, cov, rf)
        row = {"date": as_of, **dict(zip(asset_cols, w)),
               "return": stats["return"], "std": stats["std"], "sharpe": stats["sharpe"],
               "rf_coverage": rf_coverage}
        records.append(row)

    return pd.DataFrame(records).set_index("date")


if __name__ == "__main__":
    import data

    df = data.build_return_dataset(real=True)
    asset_cols = list(config.PRIMARY_ASSETS.keys())

    print("=== Full sample, real returns, 25% floor ===")
    mean, cov = annualized_mean_cov(df, asset_cols)
    rf, rf_cov = average_risk_free(df)
    print("Mean (annual):", dict(zip(asset_cols, np.round(mean * 100, 2))))
    print("Risk-free (annual):", None if rf is None else round(rf * 100, 2), "|", rf_cov)

    w_mv = min_variance(cov)
    print("Min-variance weights:", dict(zip(asset_cols, np.round(w_mv * 100, 1))))

    w_tan = tangency(mean, cov, rf)
    stats = portfolio_stats(w_tan, mean, cov, rf)
    print("Tangency weights:", dict(zip(asset_cols, np.round(w_tan * 100, 1))))
    print(f"  return={stats['return']*100:.2f}%  std={stats['std']*100:.2f}%  sharpe={stats['sharpe']:.3f}")

    print()
    print("=== 3-year rolling tangency weights (last 5 rows) ===")
    rolling = rolling_tangency_weights(df, asset_cols)
    print(rolling.tail())
