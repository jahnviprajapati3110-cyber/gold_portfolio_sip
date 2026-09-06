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


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")

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

tab_manual, tab_optimize, tab_frontier, tab_regime, tab_rolling = st.tabs(
    ["Manual Portfolio", "Optimize", "Efficient Frontier", "Regime Comparison", "Rolling Window"]
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

    m1, m2, m3 = st.columns(3)
    m1.metric("Annualized Return", f"{stats['return']*100:.2f}%")
    m2.metric("Annualized Std Dev", f"{stats['std']*100:.2f}%")
    m3.metric("Sharpe Ratio", f"{stats.get('sharpe', float('nan')):.3f}" if rf is not None else "N/A (no risk-free data)")
    st.caption(f"Risk-free rate: {'N/A' if rf is None else f'{rf*100:.2f}% (annualized)'} | coverage: {rf_coverage}")

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
            st.dataframe(comparison.style.format("{:.1%}"))

            rows = []
            for name, weights in [("Tangency", w_tan), ("Min-Variance", w_mv), ("Equal-Weight", w_eq)]:
                s = optimizer.portfolio_stats(weights, mean, cov, rf)
                rows.append({"Portfolio": name, "Return": s["return"], "Std Dev": s["std"], "Sharpe": s["sharpe"]})
            stats_df = pd.DataFrame(rows).set_index("Portfolio")
            st.dataframe(stats_df.style.format({"Return": "{:.2%}", "Std Dev": "{:.2%}", "Sharpe": "{:.3f}"}))
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
    st.dataframe(rt.style.format({
        "Risk-free (ann.)": "{:.2%}",
        **{f"{ASSET_LABELS[a]} wt": "{:.1%}" for a in asset_cols},
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
