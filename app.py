"""
app.py
------
Streamlit dashboard. Run with:  streamlit run app.py

Lets you: enter manual portfolio weights and see their return/risk/Sharpe;
click Optimize to get the constrained tangency portfolio for comparison;
view the efficient frontier; inspect how the optimal gold weight has
drifted across regimes and on a rolling basis; and swap in robustness
proxies for gold/bond.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import data
import optimizer
import risk_metrics

st.set_page_config(page_title="How Much Gold Should an Indian Investor Hold?", layout="wide")

ASSET_LABELS = {"equity": "Equity (Nifty TRI)", "bond": "Bond (G-Sec Index)", "gold": "Gold"}


# ---------------------------------------------------------------------------
# Cached data loading -- re-runs only when the floor/proxy/return-basis
# selections actually change, not on every widget interaction.
# ---------------------------------------------------------------------------

@st.cache_data
def load_dataset(gold_sheet: str, bond_sheet: str, real: bool) -> pd.DataFrame:
    asset_sheets = {"equity": config.PRIMARY_ASSETS["equity"], "bond": bond_sheet, "gold": gold_sheet}
    return data.build_return_dataset(asset_sheets=asset_sheets, real=real)


@st.cache_data
def rolling_weights(df: pd.DataFrame, asset_cols: tuple, window: int, floor: float) -> pd.DataFrame:
    return optimizer.rolling_tangency_weights(df, list(asset_cols), window_months=window, floor=floor)


def regime_table(df: pd.DataFrame, asset_cols: list, floor: float) -> pd.DataFrame:
    rows = []
    for regime in config.REGIME_ORDER:
        sub = df if regime == "Full Sample" else df[df["regime"] == regime]
        if len(sub) == 0:
            continue
        mean, cov = optimizer.annualized_mean_cov(sub, asset_cols)
        rf, rf_cov = optimizer.average_risk_free(sub)
        w_mv = optimizer.min_variance(cov, floor)
        row = {"Regime": regime, "Months": len(sub), "Risk-free (ann.)": rf, "RF coverage": rf_cov}
        if rf is not None:
            w_tan = optimizer.tangency(mean, cov, rf, floor)
            stats = optimizer.portfolio_stats(w_tan, mean, cov, rf)
            for a, w in zip(asset_cols, w_tan):
                row[f"{ASSET_LABELS[a]} wt"] = w
            row["Return"] = stats["return"]
            row["Std Dev"] = stats["std"]
            row["Sharpe"] = stats["sharpe"]
        rows.append(row)
    return pd.DataFrame(rows)


def asset_value_breakdown(weights: np.ndarray, asset_cols: list, investment_amount: float) -> pd.DataFrame:
    """Split an investment amount across assets according to a weight
    vector, for display. Returns a small DataFrame with weight (%) and
    value (currency) columns, one row per asset.
    """
    return pd.DataFrame({
        "Asset": [ASSET_LABELS[a] for a in asset_cols],
        "Weight": weights,
        "Value": weights * investment_amount,
    }).set_index("Asset")


def fmt_pct_and_value(pct: float, investment_amount: float, currency: str = "\u20b9") -> str:
    """'12.34% (\u20b91,23,400)' -- the standard display format used throughout
    this app whenever a percentage has a natural currency equivalent.
    """
    return f"{pct*100:.2f}%  ({currency}{pct*investment_amount:,.0f})"


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")

investment_amount = st.sidebar.number_input(
    "Investment amount (\u20b9)", min_value=0.0, value=100000.0, step=10000.0, format="%.2f",
    help="Every percentage figure in this app is also shown in currency terms, computed as "
         "this amount x the relevant weight or return/risk percentage. Change it any time -- "
         "nothing is recalculated from scratch, it's a straight multiplication."
)

return_basis = st.sidebar.radio("Return basis", ["Real (CPI-adjusted, Fisher)", "Nominal"], index=0)
real = return_basis.startswith("Real")

gold_choice = st.sidebar.selectbox("Gold proxy", list(config.GOLD_PROXIES.keys()), index=0)
bond_choice = st.sidebar.selectbox("Bond proxy", list(config.BOND_PROXIES.keys()), index=0)
gold_sheet = config.GOLD_PROXIES[gold_choice]
bond_sheet = config.BOND_PROXIES[bond_choice]

floor_pct = st.sidebar.slider("Minimum allocation floor per asset (%)", 0, 33, 25, step=1,
                               help="0% removes the floor entirely, showing the unconstrained "
                                    "data-driven answer. 33% forces an equal-weight portfolio.")
floor = floor_pct / 100

rolling_window = st.sidebar.slider("Rolling window (months)", 12, 60, config.DEFAULT_ROLLING_WINDOW_MONTHS, step=6)

st.sidebar.caption(
    "XAU-USD is quoted in USD/oz; selecting it as the gold proxy without an FX conversion step "
    "will give a nonsensical INR portfolio -- this demo build uses the raw USD series to keep "
    "the pipeline simple. See data.xau_usd_in_inr() for the INR-converted version."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

try:
    df = load_dataset(gold_sheet, bond_sheet, real)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

asset_cols = ["equity", "bond", "gold"]

st.title("How Much Gold Should an Indian Investor Hold?")
st.caption("Revisiting gold's hedge and diversification role in an equity-bond portfolio, 2016-2026")

st.markdown(f"**{len(df)} months loaded** ({df.index.min():%b-%Y} to {df.index.max():%b-%Y}) "
            f"| Gold proxy: **{gold_choice}** | Bond proxy: **{bond_choice}** | Basis: **{return_basis}**")

tab_manual, tab_optimize, tab_frontier, tab_regime, tab_rolling, tab_risk = st.tabs(
    ["Manual Portfolio", "Optimize", "Efficient Frontier", "Regime Comparison", "Rolling Window", "Risk Metrics"]
)

# ---------------------------------------------------------------------------
# Tab 1: manual weights
# ---------------------------------------------------------------------------

with tab_manual:
    st.subheader("Enter your own weights")
    c1, c2, c3 = st.columns(3)
    w_equity = c1.slider(ASSET_LABELS["equity"], 0, 100, 33) / 100
    w_bond = c2.slider(ASSET_LABELS["bond"], 0, 100, 34) / 100
    w_gold = c3.slider(ASSET_LABELS["gold"], 0, 100, 33) / 100
    total = w_equity + w_bond + w_gold

    if abs(total - 1.0) > 1e-9:
        st.warning(f"Weights sum to {total*100:.1f}%, not 100% -- normalizing for the calculation below.")
    w = np.array([w_equity, w_bond, w_gold])
    w = w / w.sum() if w.sum() > 0 else np.array([1 / 3] * 3)

    mean, cov = optimizer.annualized_mean_cov(df, asset_cols)
    rf, rf_coverage = optimizer.average_risk_free(df)
    stats = optimizer.portfolio_stats(w, mean, cov, rf)
    risk = risk_metrics.full_risk_report(df, asset_cols, w, rf)

    st.markdown(f"**Allocation of \u20b9{investment_amount:,.0f}**")
    breakdown = asset_value_breakdown(w, asset_cols, investment_amount)
    st.dataframe(breakdown.style.format({"Weight": "{:.1%}", "Value": "\u20b9{:,.0f}"}))

    m1, m2, m3 = st.columns(3)
    m1.metric("Annualized Return", fmt_pct_and_value(stats["return"], investment_amount))
    m2.metric("Annualized Std Dev", fmt_pct_and_value(stats["std"], investment_amount),
              help="The currency figure is the size of a typical (1 standard deviation) annual "
                   "swing in value, not a loss that's certain to happen.")
    m3.metric("Sharpe Ratio", f"{stats.get('sharpe', float('nan')):.3f}" if rf is not None else "N/A (no risk-free data)")

    m4, m5, m6 = st.columns(3)
    m4.metric("Sortino Ratio", f"{risk['sortino_ratio']:.3f}" if rf is not None else "N/A (no risk-free data)",
              help="Like Sharpe, but only penalizes downside moves -- a fairer measure when a "
                   "return series isn't symmetric (gold and equity rarely are).")
    m5.metric("Max Drawdown", fmt_pct_and_value(risk["max_drawdown"], investment_amount),
              help="The worst peak-to-trough decline actually experienced in this sample, and what "
                   "that decline would have meant in currency terms on this investment amount.")
    m6.metric("Calmar Ratio", f"{risk['calmar_ratio']:.3f}",
              help="Annualized return divided by |max drawdown| -- a 'worst realistic case' risk-adjusted measure.")

    st.caption(f"Risk-free rate: {'N/A' if rf is None else f'{rf*100:.2f}% (annualized)'} | coverage: {rf_coverage}")

    with st.expander("More risk detail: downside deviation, VaR, CVaR"):
        d1, d2, d3 = st.columns(3)
        d1.metric("Downside Deviation", fmt_pct_and_value(risk["downside_deviation_annual"], investment_amount))
        d2.metric("Historical VaR (95%, monthly)", fmt_pct_and_value(risk["var_95"], investment_amount),
                  help="The loss you would not expect to exceed in 95% of months, based on this "
                       "sample's actual history -- shown as a monthly currency loss, not annual.")
        d3.metric("Historical CVaR (95%, monthly)", fmt_pct_and_value(risk["cvar_95"], investment_amount),
                  help="The AVERAGE loss in the worst 5% of months -- how bad the bad tail actually "
                       "is, not just where it starts. Also a monthly figure.")

    st.markdown("**Correlation matrix**")
    st.dataframe(df[asset_cols].corr().rename(index=ASSET_LABELS, columns=ASSET_LABELS).style.format("{:.2f}"))

    st.markdown("**Covariance matrix (annualized)**")
    cov_df = pd.DataFrame(cov, index=asset_cols, columns=asset_cols).rename(index=ASSET_LABELS, columns=ASSET_LABELS)
    st.dataframe(cov_df.style.format("{:.4f}"))

# ---------------------------------------------------------------------------
# Tab 2: optimize + compare
# ---------------------------------------------------------------------------

with tab_optimize:
    st.subheader("Optimize Portfolio")
    if st.button("Optimize Portfolio (max Sharpe, full sample)", type="primary"):
        mean, cov = optimizer.annualized_mean_cov(df, asset_cols)
        rf, rf_coverage = optimizer.average_risk_free(df)

        if rf is None:
            st.error("No risk-free data available for this sample -- cannot compute a tangency portfolio.")
        else:
            w_tan = optimizer.tangency(mean, cov, rf, floor)
            w_mv = optimizer.min_variance(cov, floor)
            w_eq = optimizer.equal_weight_portfolio(mean, cov, rf)["weights"]

            comparison = pd.DataFrame({
                "Manual (from the Manual Portfolio tab)": w,
                "Tangency (max Sharpe)": w_tan,
                "Min-Variance": w_mv,
                "Equal-Weight": w_eq,
            }, index=[ASSET_LABELS[a] for a in asset_cols])
            st.markdown(f"**Floor applied: {floor_pct}% per asset**")
            st.markdown("Weights (%)")
            st.dataframe(comparison.style.format("{:.1%}"))
            st.markdown(f"Allocation of \u20b9{investment_amount:,.0f}")
            st.dataframe((comparison * investment_amount).style.format("\u20b9{:,.0f}"))

            rows = []
            for name, weights in [("Manual", w), ("Tangency", w_tan), ("Min-Variance", w_mv), ("Equal-Weight", w_eq)]:
                s = optimizer.portfolio_stats(weights, mean, cov, rf)
                r = risk_metrics.full_risk_report(df, asset_cols, weights, rf)
                rows.append({
                    "Portfolio": name,
                    "Return": s["return"], "Return (\u20b9/yr)": s["return"] * investment_amount,
                    "Std Dev": s["std"], "Std Dev (\u20b9/yr)": s["std"] * investment_amount,
                    "Sharpe": s["sharpe"],
                    "Max Drawdown": r["max_drawdown"], "Max Drawdown (\u20b9)": r["max_drawdown"] * investment_amount,
                })
            stats_df = pd.DataFrame(rows).set_index("Portfolio")
            st.dataframe(stats_df.style.format({
                "Return": "{:.2%}", "Return (\u20b9/yr)": "\u20b9{:,.0f}",
                "Std Dev": "{:.2%}", "Std Dev (\u20b9/yr)": "\u20b9{:,.0f}",
                "Sharpe": "{:.3f}",
                "Max Drawdown": "{:.2%}", "Max Drawdown (\u20b9)": "\u20b9{:,.0f}",
            }))
    else:
        st.info("Click the button to run the constrained optimization (SLSQP, full sample).")

# ---------------------------------------------------------------------------
# Tab 3: efficient frontier
# ---------------------------------------------------------------------------

with tab_frontier:
    st.subheader("Efficient Frontier (full sample)")
    mean, cov = optimizer.annualized_mean_cov(df, asset_cols)
    frontier = optimizer.efficient_frontier(mean, cov, floor)
    rf, _ = optimizer.average_risk_free(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[pt["std"] * 100 for pt in frontier],
        y=[pt["target_return"] * 100 for pt in frontier],
        mode="lines+markers", name="Efficient Frontier",
    ))
    if rf is not None:
        w_tan = optimizer.tangency(mean, cov, rf, floor)
        tan_stats = optimizer.portfolio_stats(w_tan, mean, cov, rf)
        fig.add_trace(go.Scatter(
            x=[tan_stats["std"] * 100], y=[tan_stats["return"] * 100],
            mode="markers", marker=dict(size=14, symbol="star"), name="Tangency Portfolio",
        ))
    fig.update_layout(xaxis_title="Annualized Std Dev (%)", yaxis_title="Annualized Return (%)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show frontier weights"):
        ft_df = pd.DataFrame([
            {"Target Return": pt["target_return"], "Std Dev": pt["std"],
             **{ASSET_LABELS[a]: w for a, w in zip(asset_cols, pt["weights"])}}
            for pt in frontier
        ])
        st.dataframe(ft_df.style.format({"Target Return": "{:.2%}", "Std Dev": "{:.2%}",
                                          **{ASSET_LABELS[a]: "{:.1%}" for a in asset_cols}}))

# ---------------------------------------------------------------------------
# Tab 4: regime comparison -- the headline result
# ---------------------------------------------------------------------------

with tab_regime:
    st.subheader("Optimal Gold Weight by Regime")
    rt = regime_table(df, asset_cols, floor)
    for a in asset_cols:
        wt_col = f"{ASSET_LABELS[a]} wt"
        if wt_col in rt.columns:
            rt[f"{ASSET_LABELS[a]} value"] = rt[wt_col] * investment_amount
    st.caption(f"'value' columns show the allocation on \u20b9{investment_amount:,.0f}.")
    st.dataframe(rt.style.format({
        "Risk-free (ann.)": "{:.2%}",
        **{f"{ASSET_LABELS[a]} wt": "{:.1%}" for a in asset_cols},
        **{f"{ASSET_LABELS[a]} value": "\u20b9{:,.0f}" for a in asset_cols},
        "Return": "{:.2%}", "Std Dev": "{:.2%}", "Sharpe": "{:.3f}",
    }))

    if floor_pct == config.DEFAULT_FLOOR * 100:
        st.warning(
            "Try dragging the floor slider in the sidebar down toward 0% -- with the 25% floor, "
            "gold's weight may look regime-stable simply because the floor is binding, not because "
            "the underlying data says so. Compare against the unconstrained (0% floor) result to "
            "see gold's genuine, data-driven attractiveness in each regime."
        )

    gold_col = f"{ASSET_LABELS['gold']} wt"
    if gold_col in rt.columns:
        fig = go.Figure(go.Bar(x=rt["Regime"], y=rt[gold_col] * 100))
        fig.update_layout(yaxis_title="Optimal Gold Weight (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: rolling window
# ---------------------------------------------------------------------------

with tab_rolling:
    st.subheader(f"{rolling_window}-Month Rolling Tangency Weights")
    rolling = rolling_weights(df, tuple(asset_cols), rolling_window, floor)

    fig = go.Figure()
    for a in asset_cols:
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling[a] * 100, mode="lines", name=ASSET_LABELS[a]))
    fig.update_layout(yaxis_title="Optimal weight (%)", xaxis_title="Date", height=500,
                       title="Rolling optimal weights (this is likely your single most important chart)")
    st.plotly_chart(fig, use_container_width=True)

    missing = rolling["return"].isna().sum()
    if missing:
        st.caption(f"{missing} of {len(rolling)} rolling windows had no usable risk-free data "
                   f"and are shown as gaps -- a real limitation of the T-Bill series' coverage, "
                   f"not a computation error.")

    with st.expander("Show rolling window data"):
        st.dataframe(rolling.style.format({a: "{:.1%}" for a in asset_cols} |
                                           {"return": "{:.2%}", "std": "{:.2%}", "sharpe": "{:.3f}"}))

# ---------------------------------------------------------------------------
# Tab 6: risk metrics (Sortino, VaR/CVaR, Jensen's alpha, Monte Carlo)
# ---------------------------------------------------------------------------

with tab_risk:
    st.subheader("Risk Metrics Beyond Standard Deviation")

    mean, cov = optimizer.annualized_mean_cov(df, asset_cols)
    rf, rf_coverage = optimizer.average_risk_free(df)

    portfolio_choice = st.selectbox(
        "Portfolio to analyze",
        ["Tangency (max Sharpe)", "Minimum-Variance", "Equal-Weight", "Custom"],
    )

    if portfolio_choice == "Tangency (max Sharpe)" and rf is not None:
        w_selected = optimizer.tangency(mean, cov, rf, floor)
    elif portfolio_choice == "Minimum-Variance":
        w_selected = optimizer.min_variance(cov, floor)
    elif portfolio_choice == "Equal-Weight":
        w_selected = np.array([1 / 3] * 3)
    elif portfolio_choice == "Custom":
        c1, c2, c3 = st.columns(3)
        cw_equity = c1.slider(f"{ASSET_LABELS['equity']} (risk tab)", 0, 100, 33) / 100
        cw_bond = c2.slider(f"{ASSET_LABELS['bond']} (risk tab)", 0, 100, 34) / 100
        cw_gold = c3.slider(f"{ASSET_LABELS['gold']} (risk tab)", 0, 100, 33) / 100
        raw = np.array([cw_equity, cw_bond, cw_gold])
        w_selected = raw / raw.sum() if raw.sum() > 0 else np.array([1 / 3] * 3)
    else:
        st.warning("No risk-free data available for a tangency portfolio in this sample -- "
                   "showing Equal-Weight instead.")
        w_selected = np.array([1 / 3] * 3)

    st.markdown("**Weights being analyzed:** " +
                " / ".join(f"{ASSET_LABELS[a]} {w*100:.0f}%" for a, w in zip(asset_cols, w_selected)))

    st.markdown(f"**Allocation of \u20b9{investment_amount:,.0f}**")
    breakdown = asset_value_breakdown(w_selected, asset_cols, investment_amount)
    st.dataframe(breakdown.style.format({"Weight": "{:.1%}", "Value": "\u20b9{:,.0f}"}))

    report = risk_metrics.full_risk_report(df, asset_cols, w_selected, rf, market_col="equity")

    st.markdown("#### Standard vs. downside-only risk")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Std Dev (annual)", fmt_pct_and_value(report["std_annual"], investment_amount))
    c2.metric("Downside Deviation", fmt_pct_and_value(report["downside_deviation_annual"], investment_amount),
              help="Same idea as std dev, but only counts months below the target return -- "
                   "upside surprises don't count against you here.")
    c3.metric("Sharpe Ratio", f"{report.get('sharpe_ratio', float('nan')):.3f}" if rf is not None else "N/A")
    c4.metric("Sortino Ratio", f"{report.get('sortino_ratio', float('nan')):.3f}" if rf is not None else "N/A",
              help="Like Sharpe, but only penalizes downside volatility -- typically higher than "
                   "Sharpe when a portfolio's losses are smaller/rarer than its gains.")

    st.markdown("#### Tail risk")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Drawdown", fmt_pct_and_value(report["max_drawdown"], investment_amount),
              help="The worst peak-to-trough decline actually experienced in this sample, in both "
                   "percentage and currency terms on your investment amount.")
    c2.metric("Historical VaR (95%)", fmt_pct_and_value(report["var_95"], investment_amount),
              help="The monthly loss you would not expect to exceed in 95% of months, based on "
                   "what actually happened -- makes no assumption about the shape of the return "
                   "distribution. Currency figure is a MONTHLY loss, not annual.")
    c3.metric("Historical CVaR (95%)", fmt_pct_and_value(report["cvar_95"], investment_amount),
              help="The AVERAGE loss in the worst 5% of months -- always >= VaR, answers 'if it's "
                   "bad, how bad on average' rather than just where the bad tail starts. Also monthly.")
    c4.metric("Calmar Ratio", f"{report['calmar_ratio']:.3f}",
              help="Annualized return divided by the worst drawdown -- a 'worst realistic case' "
                   "risk-adjusted return measure popular in fund management.")

    port_returns_for_var = risk_metrics.portfolio_return_series(df, asset_cols, w_selected)
    parametric_var_95 = risk_metrics.parametric_var(pd.Series(port_returns_for_var))
    st.caption(
        f"For comparison, parametric VaR (assumes a normal distribution) at 95% confidence: "
        f"**{parametric_var_95*100:.2f}% (\u20b9{parametric_var_95*investment_amount:,.0f})** monthly loss. "
        f"A large gap between this and the historical VaR above means actual returns have fatter "
        f"tails than a normal distribution predicts -- worth noting in your write-up rather than "
        f"picking whichever number looks better."
    )

    if rf is not None:
        st.markdown("#### Jensen's Alpha (vs. equity market)")
        port_returns = risk_metrics.portfolio_return_series(df, asset_cols, w_selected)
        ja = risk_metrics.jensen_alpha(port_returns, df["equity"].values, rf)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beta (vs equity)", f"{ja['beta']:.3f}")
        c2.metric("CAPM Expected Return", fmt_pct_and_value(ja["expected_return"], investment_amount))
        c3.metric("Actual Return", fmt_pct_and_value(ja["actual_return"], investment_amount))
        c4.metric("Jensen's Alpha", fmt_pct_and_value(ja["alpha"], investment_amount),
                  help="Actual return minus what CAPM predicts given this portfolio's beta to "
                       "equities, both as a percentage and as the currency value of that extra "
                       "return per year. Positive = genuine outperformance beyond just diluted "
                       "equity risk.")
        st.caption(
            "Market benchmark used: Nifty TRI (equity). This measures whether the equity-gold-bond "
            "portfolio earns more than you'd expect from its equity-risk exposure alone -- a low "
            "beta with a strongly positive alpha (as is typical here) is exactly what a genuine "
            "diversification benefit looks like in CAPM terms."
        )
    else:
        st.info("Jensen's alpha requires a risk-free rate, which isn't available for this sample.")

    st.markdown("---")
    st.markdown("#### Monte Carlo Simulation")
    st.caption(
        "Bootstrap resampling: draws actual historical monthly return VECTORS (all three assets "
        "together, same month) with replacement, preserving the real joint distribution -- fat "
        "tails, skew, and correlations -- rather than assuming returns are normally distributed."
    )
    mc1, mc2, mc3 = st.columns(3)
    horizon_months = mc1.slider("Horizon (months)", 1, 60, 12)
    n_sims = mc2.select_slider("Number of simulations", options=[1000, 2500, 5000, 10000], value=5000)
    mc_method = mc3.radio("Method", ["bootstrap", "parametric"], horizontal=True)

    if st.button("Run Monte Carlo Simulation"):
        mc = risk_metrics.monte_carlo_simulation(
            df, asset_cols, w_selected, n_months=horizon_months, n_sims=n_sims,
            method=mc_method, seed=42,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probability of Loss", f"{mc['probability_of_loss']*100:.1f}%")
        m2.metric("Median Return", fmt_pct_and_value(mc["median_return"], investment_amount))
        m3.metric("Simulated VaR (95%)", fmt_pct_and_value(mc["var_95"], investment_amount),
                  help=f"Currency figure is the loss over the full {horizon_months}-month horizon, not monthly.")
        m4.metric("Simulated CVaR (95%)", fmt_pct_and_value(mc["cvar_95"], investment_amount),
                  help=f"Currency figure is the loss over the full {horizon_months}-month horizon, not monthly.")

        # wealth_paths from risk_metrics.monte_carlo_simulation is a cumulative
        # GROWTH FACTOR starting at 1.0 (e.g. 1.05 = +5%), not currency -- scale
        # by the actual investment amount to show real portfolio value in rupees.
        wealth_paths_value = mc["wealth_paths"] * investment_amount
        months_axis = np.arange(1, horizon_months + 1)
        p5 = np.percentile(wealth_paths_value, 5, axis=0)
        p25 = np.percentile(wealth_paths_value, 25, axis=0)
        p50 = np.percentile(wealth_paths_value, 50, axis=0)
        p75 = np.percentile(wealth_paths_value, 75, axis=0)
        p95 = np.percentile(wealth_paths_value, 95, axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months_axis, y=p95, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=months_axis, y=p5, fill="tonexty", line=dict(width=0),
                                  name="5th-95th percentile", fillcolor="rgba(99,110,250,0.15)"))
        fig.add_trace(go.Scatter(x=months_axis, y=p75, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=months_axis, y=p25, fill="tonexty", line=dict(width=0),
                                  name="25th-75th percentile", fillcolor="rgba(99,110,250,0.3)"))
        fig.add_trace(go.Scatter(x=months_axis, y=p50, line=dict(color="rgb(99,110,250)", width=2),
                                  name="Median path"))
        fig.add_hline(y=investment_amount, line_dash="dot", line_color="gray",
                      annotation_text="Initial investment")
        fig.update_layout(xaxis_title="Month", yaxis_title=f"Portfolio value (\u20b9, starting at {investment_amount:,.0f})",
                           height=450, title=f"{n_sims:,} simulated {horizon_months}-month paths ({mc_method})")
        st.plotly_chart(fig, use_container_width=True)

        final_values = investment_amount * (1 + mc["final_returns"])
        fig2 = go.Figure(go.Histogram(x=final_values, nbinsx=60))
        fig2.add_vline(x=investment_amount, line_dash="dot", line_color="gray",
                       annotation_text="Initial investment")
        fig2.update_layout(xaxis_title=f"Portfolio value after {horizon_months} months (\u20b9)", yaxis_title="Count",
                            height=350, title="Distribution of simulated outcomes")
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            f"Value at key percentiles after {horizon_months} months: "
            f"5th percentile \u20b9{np.percentile(final_values,5):,.0f}  |  "
            f"median \u20b9{np.percentile(final_values,50):,.0f}  |  "
            f"95th percentile \u20b9{np.percentile(final_values,95):,.0f}  "
            f"(started at \u20b9{investment_amount:,.0f})."
        )
    else:
        st.info("Click the button to run the simulation.")
