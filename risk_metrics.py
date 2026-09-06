"""
risk_metrics.py
---------------
Risk measures beyond plain standard deviation:

- Downside deviation / Sortino ratio: only penalizes returns below a target
  (usually 0), unlike std dev which treats a big gain and a big loss as
  equally "risky". Generally a fairer risk-adjusted measure for return
  series that aren't symmetric (gold and equity returns rarely are).
- Maximum drawdown: the worst peak-to-trough decline actually experienced
  in the sample -- answers "how bad did it get if you'd lived through it",
  which Sharpe/Sortino (both averages) can hide.
- Historical VaR / CVaR: how much you could lose in a bad month at a given
  confidence level, and the average loss in the worst-case tail beyond that.
- Monte Carlo simulation: simulate many possible future paths for a given
  portfolio, either by bootstrapping actual historical monthly return
  vectors (preserves the real, non-normal joint distribution and
  correlations) or by drawing from a fitted multivariate normal
  (parametric -- faster, but assumes returns are normally distributed,
  which understates tail risk for real asset returns).

Nothing here does optimization -- see optimizer.py for that. This module
only evaluates a portfolio (a fixed weight vector) that's already been
chosen, whether manually or by the optimizer.
"""

import numpy as np
import pandas as pd


def portfolio_return_series(returns_df: pd.DataFrame, asset_cols: list, weights) -> pd.Series:
    """Combine per-asset monthly returns into a single portfolio return
    series for a given weight vector. This is the input every function
    below expects.
    """
    weights = np.asarray(weights)
    return (returns_df[asset_cols].values @ weights)


def downside_deviation(monthly_returns: np.ndarray, target: float = 0.0) -> float:
    """Annualized downside deviation: like std dev, but only counts
    returns below `target` (default 0, i.e. any month with a loss).
    """
    monthly_returns = np.asarray(monthly_returns)
    shortfall = np.minimum(monthly_returns - target, 0)
    if len(shortfall) == 0:
        return 0.0
    monthly_dd = np.sqrt((shortfall ** 2).mean())
    return monthly_dd * np.sqrt(12)


def sortino_ratio(monthly_returns: np.ndarray, risk_free_annual: float, target: float = 0.0) -> float:
    """(annualized return - risk-free) / annualized downside deviation.
    Same idea as Sharpe, but the denominator only counts downside moves.
    Returns NaN if there's no downside at all in the sample (undefined,
    not infinite -- don't report an infinite Sortino ratio).
    """
    monthly_returns = np.asarray(monthly_returns)
    mean_annual = monthly_returns.mean() * 12
    dd_annual = downside_deviation(monthly_returns, target)
    if dd_annual == 0:
        return np.nan
    return (mean_annual - risk_free_annual) / dd_annual


def max_drawdown(monthly_returns: np.ndarray) -> dict:
    """Worst peak-to-trough decline in cumulative wealth over the sample.
    Returns the drawdown as a negative fraction (e.g. -0.23 = a 23% decline)
    plus the full drawdown series so it can be charted.
    """
    monthly_returns = np.asarray(monthly_returns)
    wealth = np.cumprod(1 + monthly_returns)
    running_peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_peak) / running_peak
    return {"max_drawdown": float(drawdown.min()), "drawdown_series": drawdown, "wealth_series": wealth}


def historical_var(monthly_returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical (non-parametric) Value at Risk: the loss you would not
    expect to exceed in `confidence`% of months, based on what actually
    happened in this sample. Returned as a POSITIVE number representing
    the size of the loss (e.g. 0.08 means an 8% loss).
    """
    monthly_returns = np.asarray(monthly_returns)
    return float(-np.percentile(monthly_returns, (1 - confidence) * 100))


def historical_cvar(monthly_returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall: the AVERAGE loss in the worst
    (1-confidence)% of months -- answers "if things are bad, how bad", a
    more complete tail-risk picture than VaR alone (which only marks the
    boundary of the bad tail, not how bad the tail itself is).
    """
    monthly_returns = np.asarray(monthly_returns)
    threshold = np.percentile(monthly_returns, (1 - confidence) * 100)
    tail = monthly_returns[monthly_returns <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-tail.mean())


def parametric_var(monthly_returns: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric (variance-covariance) VaR: assumes returns are normally
    distributed, using only the mean and std dev rather than the empirical
    tail. Faster and smoother than historical VaR, but understates risk if
    actual returns have fatter tails than normal -- a real concern with
    only ~40-50 months per regime here. Compare both and report the gap.
    """
    from scipy.stats import norm
    monthly_returns = np.asarray(monthly_returns)
    mu, sigma = monthly_returns.mean(), monthly_returns.std(ddof=1)
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def calmar_ratio(annual_return: float, max_dd: float) -> float:
    """Annualized return / |max drawdown|. Popular in fund management as
    a "worst realistic case" risk-adjusted return measure.
    """
    if max_dd == 0:
        return np.nan
    return annual_return / abs(max_dd)


def beta(asset_returns, market_returns) -> float:
    """Beta of a portfolio/asset relative to a market benchmark -- how much
    it moves per unit of market move, computed on whatever months both
    series share. Beta > 1 amplifies market moves; beta < 1 dampens them;
    beta near 0 means largely uncorrelated with the market.
    """
    asset_returns = pd.Series(np.asarray(asset_returns))
    market_returns = pd.Series(np.asarray(market_returns)).reset_index(drop=True)
    asset_returns = asset_returns.reset_index(drop=True)
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    aligned.columns = ["asset", "market"]
    cov = aligned["asset"].cov(aligned["market"])
    var_m = aligned["market"].var()
    return cov / var_m if var_m != 0 else np.nan


def jensen_alpha(portfolio_returns, market_returns, risk_free_annual: float) -> dict:
    """
    Jensen's alpha: the portfolio's actual annualized return, minus what
    CAPM says it SHOULD have earned given only its beta exposure to the
    market. Positive alpha = genuine outperformance beyond just carrying
    market risk; negative = it underperformed what its beta alone predicts.

    market_returns: typically the equity series (Nifty TRI) here, since
    that's the closest thing to "the market" in this project. This answers:
    does diversifying into gold and bonds earn more than you'd expect from
    a portfolio that's simply a diluted, lower-beta version of equities?
    """
    b = beta(portfolio_returns, market_returns)
    port = pd.Series(np.asarray(portfolio_returns)).reset_index(drop=True)
    mkt = pd.Series(np.asarray(market_returns)).reset_index(drop=True)
    aligned = pd.concat([port, mkt], axis=1).dropna()
    aligned.columns = ["port", "mkt"]
    port_annual = aligned["port"].mean() * 12
    mkt_annual = aligned["mkt"].mean() * 12
    expected_return = risk_free_annual + b * (mkt_annual - risk_free_annual)
    alpha = port_annual - expected_return
    return {"alpha": alpha, "beta": b, "expected_return": expected_return,
            "actual_return": port_annual, "market_return": mkt_annual}


def full_risk_report(returns_df: pd.DataFrame, asset_cols: list, weights,
                      risk_free_annual: float = None, confidence: float = 0.95,
                      market_col: str = None) -> dict:
    """Convenience wrapper: every metric above, for one portfolio, in one call.

    market_col: if given (e.g. "equity"), also computes beta and Jensen's
    alpha against that column of returns_df as the market benchmark.
    """
    port = portfolio_return_series(returns_df, asset_cols, weights)
    mean_annual = port.mean() * 12
    std_annual = port.std(ddof=1) * np.sqrt(12)
    dd = max_drawdown(port)

    report = {
        "return_annual": mean_annual,
        "std_annual": std_annual,
        "downside_deviation_annual": downside_deviation(port),
        "max_drawdown": dd["max_drawdown"],
        "var_95": historical_var(port, confidence),
        "cvar_95": historical_cvar(port, confidence),
        "calmar_ratio": calmar_ratio(mean_annual, dd["max_drawdown"]),
    }
    if risk_free_annual is not None:
        report["sharpe_ratio"] = (mean_annual - risk_free_annual) / std_annual
        report["sortino_ratio"] = sortino_ratio(port, risk_free_annual)
        if market_col is not None:
            ja = jensen_alpha(port, returns_df[market_col].values, risk_free_annual)
            report["jensen_alpha"] = ja["alpha"]
            report["beta"] = ja["beta"]
    return report


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def monte_carlo_simulation(returns_df: pd.DataFrame, asset_cols: list, weights,
                            n_months: int = 12, n_sims: int = 5000,
                            method: str = "bootstrap", seed: int = None) -> dict:
    """
    Simulate `n_sims` possible future paths of `n_months` months each for a
    fixed-weight portfolio, using one of two methods:

    "bootstrap" (default, recommended): resample actual historical monthly
    return VECTORS (all three assets together, same month) with replacement.
    This preserves the real joint distribution -- fat tails, skew, and the
    actual correlation structure -- rather than assuming returns are
    normally distributed, which they usually aren't (a single bad month can
    hit all three assets' tails simultaneously in ways a normal distribution
    understates).

    "parametric": draw from a multivariate normal fitted to the historical
    mean vector and covariance matrix. Faster and smoother-looking, but
    will understate the odds of extreme joint moves (crashes) compared to
    bootstrap, since real returns have fatter tails than a normal
    distribution assumes.

    Returns a dict with the full simulated wealth paths (for a fan chart),
    the distribution of final cumulative returns, and summary statistics.
    """
    rng = np.random.default_rng(seed)
    R = returns_df[asset_cols].values
    weights = np.asarray(weights)

    if method == "bootstrap":
        n_hist = len(R)
        idx = rng.integers(0, n_hist, size=(n_sims, n_months))
        sampled = R[idx]  # (n_sims, n_months, n_assets)
    elif method == "parametric":
        mean_monthly = R.mean(axis=0)
        cov_monthly = np.cov(R, rowvar=False, ddof=1)
        sampled = rng.multivariate_normal(mean_monthly, cov_monthly, size=(n_sims, n_months))
    else:
        raise ValueError(f"Unknown method '{method}': use 'bootstrap' or 'parametric'")

    port_monthly = sampled @ weights          # (n_sims, n_months)
    wealth_paths = np.cumprod(1 + port_monthly, axis=1)  # (n_sims, n_months)
    final_returns = wealth_paths[:, -1] - 1

    var_95 = -np.percentile(final_returns, 5)
    tail = final_returns[final_returns <= np.percentile(final_returns, 5)]

    return {
        "method": method,
        "n_months": n_months,
        "n_sims": n_sims,
        "wealth_paths": wealth_paths,
        "final_returns": final_returns,
        "probability_of_loss": float((final_returns < 0).mean()),
        "median_return": float(np.median(final_returns)),
        "p5_return": float(np.percentile(final_returns, 5)),
        "p25_return": float(np.percentile(final_returns, 25)),
        "p75_return": float(np.percentile(final_returns, 75)),
        "p95_return": float(np.percentile(final_returns, 95)),
        "var_95": float(var_95),
        "cvar_95": float(-tail.mean()) if len(tail) else float(var_95),
    }


if __name__ == "__main__":
    import config
    import data
    import optimizer

    df = data.build_return_dataset(real=True)
    asset_cols = list(config.PRIMARY_ASSETS.keys())
    mean, cov = optimizer.annualized_mean_cov(df, asset_cols)
    rf, _ = optimizer.average_risk_free(df)
    w_tan = optimizer.tangency(mean, cov, rf, config.DEFAULT_FLOOR)

    print("=== Full risk report, full-sample tangency portfolio ===")
    report = full_risk_report(df, asset_cols, w_tan, rf, market_col="equity")
    pct_keys = {"return_annual", "std_annual", "downside_deviation_annual", "max_drawdown",
                "var_95", "cvar_95", "jensen_alpha"}
    for k, v in report.items():
        print(f"  {k}: {v*100:.2f}%" if k in pct_keys else f"  {k}: {v:.3f}")

    print()
    print("=== Jensen's alpha detail (vs equity market) ===")
    port_returns = portfolio_return_series(df, asset_cols, w_tan)
    ja = jensen_alpha(port_returns, df["equity"].values, rf)
    print(f"  beta={ja['beta']:.3f}  actual={ja['actual_return']*100:.2f}%  "
          f"expected(CAPM)={ja['expected_return']*100:.2f}%  alpha={ja['alpha']*100:.2f}%  "
          f"market={ja['market_return']*100:.2f}%")

    print()
    print("=== Monte Carlo, 12-month horizon, bootstrap ===")
    mc = monte_carlo_simulation(df, asset_cols, w_tan, n_months=12, n_sims=5000, method="bootstrap", seed=42)
    print(f"  Probability of loss: {mc['probability_of_loss']*100:.1f}%")
    print(f"  Median return: {mc['median_return']*100:.2f}%")
    print(f"  5th-95th percentile: {mc['p5_return']*100:.2f}% to {mc['p95_return']*100:.2f}%")
    print(f"  VaR 95%: {mc['var_95']*100:.2f}%   CVaR 95%: {mc['cvar_95']*100:.2f}%")
