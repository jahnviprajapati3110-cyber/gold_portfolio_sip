"""
data.py
-------
Everything that turns the raw Excel workbook into clean, aligned monthly
return series: nominal returns, CPI-adjusted real returns (Fisher method),
and the risk-free rate. Nothing in this file does any optimization -- that
lives in optimizer.py. This file's only job is to produce trustworthy
pandas Series/DataFrames that optimizer.py and app.py can consume.
"""

import pandas as pd
import numpy as np
from datetime import datetime

import config


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def load_price_series(sheet_name: str, path=None) -> pd.Series:
    """Load a single 'Date, Close' sheet as a pandas Series indexed by
    month-end Timestamp, sorted ascending. This is the only function that
    touches the raw Excel file for price/index sheets.
    """
    path = path or config.RAW_DATA_PATH
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(subset=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    s = pd.Series(df["Close"].values, index=df["Date"].values, name=sheet_name)
    # normalize the index to (year, month) so sheets with slightly different
    # "last trading day" dates (e.g. 29th vs 31st) still align on the same month
    s.index = s.index.to_period("M").to_timestamp("M")
    return s


def load_cpi(path=None) -> pd.DataFrame:
    """Load the CPI sheet: Date, Base Year, CPI Index, YoY Inflation (%)."""
    path = path or config.RAW_DATA_PATH
    df = pd.read_excel(path, sheet_name=config.CPI_SHEET)
    df["Date"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp("M")
    return df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def nominal_returns(price_series: pd.Series) -> pd.Series:
    """Simple month-over-month return: (P_t / P_t-1) - 1."""
    return price_series.pct_change().dropna()


def monthly_inflation(cpi_df: pd.DataFrame) -> pd.Series:
    """
    Month-over-month CPI inflation, handling the base-year rebasing correctly.

    For any two consecutive months on the SAME base year, this is the plain
    index ratio: Index_t / Index_t-1 - 1.

    For the single month where the base year changes (see
    config.CPI_REBASE_BOUNDARY), the index levels are not comparable, so
    that one month instead uses its own published year-on-year inflation
    rate, de-annualized: (1 + YoY)^(1/12) - 1.
    """
    cpi_df = cpi_df.set_index("Date").sort_index()
    index_vals = cpi_df["CPI Index"]
    base_year = cpi_df["Base Year"]
    yoy = cpi_df["YoY Inflation (%)"]

    out = {}
    dates = index_vals.index
    for i in range(1, len(dates)):
        t, t_prev = dates[i], dates[i - 1]
        if pd.isna(index_vals[t]) or pd.isna(index_vals[t_prev]):
            continue
        same_base = base_year[t] == base_year[t_prev]
        if same_base:
            out[t] = index_vals[t] / index_vals[t_prev] - 1
        elif not pd.isna(yoy[t]):
            out[t] = (1 + yoy[t]) ** (1 / 12) - 1
        # else: no usable data for this month, leave it out
    return pd.Series(out).sort_index()


def real_returns_fisher(nominal: pd.Series, inflation: pd.Series) -> pd.Series:
    """Fisher-exact real return: (1+nominal)/(1+inflation) - 1, aligned on
    the dates both series share.
    """
    aligned = pd.concat([nominal, inflation], axis=1, join="inner")
    aligned.columns = ["nominal", "inflation"]
    return ((1 + aligned["nominal"]) / (1 + aligned["inflation"]) - 1).rename("real_return")


def real_returns_approx(nominal: pd.Series, inflation: pd.Series) -> pd.Series:
    """Quick approximation: nominal - inflation. Kept for comparison only;
    prefer real_returns_fisher for anything reported.
    """
    aligned = pd.concat([nominal, inflation], axis=1, join="inner")
    aligned.columns = ["nominal", "inflation"]
    return (aligned["nominal"] - aligned["inflation"]).rename("real_return_approx")


# ---------------------------------------------------------------------------
# Regime tagging
# ---------------------------------------------------------------------------

def phase_of(date: pd.Timestamp) -> str:
    if date <= config.REGIME_BOUNDARIES["Pre-Covid"]:
        return "Pre-Covid"
    if date <= config.REGIME_BOUNDARIES["Covid"]:
        return "Covid"
    return "Post-Covid"


def add_regime_column(df: pd.DataFrame, date_col=None) -> pd.DataFrame:
    """Add a 'regime' column based on the DataFrame's index (or date_col)."""
    df = df.copy()
    dates = df.index if date_col is None else df[date_col]
    df["regime"] = [phase_of(d) for d in dates]
    return df


# ---------------------------------------------------------------------------
# Dataset assembly -- the main entry point app.py and optimizer.py should use
# ---------------------------------------------------------------------------

def build_return_dataset(asset_sheets: dict = None, real: bool = True, path=None) -> pd.DataFrame:
    """
    Build a single aligned DataFrame of monthly returns for the given assets
    (default: the three primary assets from config.PRIMARY_ASSETS), plus a
    'regime' column and a 'risk_free' column from the T-Bill sheet.

    asset_sheets: dict like {"equity": "Nifty TRI", "gold": "MCX Gold", ...}
                  -- pass this to swap in a robustness proxy for one asset.
    real: if True, columns are Fisher real returns; if False, nominal returns.

    Returns a DataFrame indexed by month-end date with columns
    [equity, bond, gold, risk_free, regime] (column names follow the keys
    of asset_sheets), inner-joined so every row has complete data.
    """
    asset_sheets = asset_sheets or config.PRIMARY_ASSETS
    cpi_df = load_cpi(path)
    inflation = monthly_inflation(cpi_df)

    cols = {}
    for key, sheet_name in asset_sheets.items():
        prices = load_price_series(sheet_name, path)
        nom = nominal_returns(prices)
        cols[key] = real_returns_fisher(nom, inflation) if real else nom

    rf_prices = load_price_series(config.RISK_FREE_SHEET, path)
    rf_nom = nominal_returns(rf_prices)
    cols["risk_free"] = real_returns_fisher(rf_nom, inflation) if real else rf_nom

    df = pd.DataFrame(cols)
    # asset columns must all be present for a valid row; risk_free may be
    # NaN (T-Bill has a shorter history) without invalidating the row
    asset_cols = list(asset_sheets.keys())
    df = df.dropna(subset=asset_cols, how="any")
    df = add_regime_column(df)
    return df


def xau_usd_in_inr(path=None) -> pd.Series:
    """Convenience helper: XAU-USD is quoted in USD/oz. Multiply by the
    USD-INR FX rate (same month) to get an INR-denominated gold price
    series comparable to MCX Gold / the Gold ETFs, before computing returns.
    """
    xau = load_price_series("XAU-USD", path)
    fx = load_price_series(config.FX_SHEET, path)
    aligned = pd.concat([xau, fx], axis=1, join="inner")
    aligned.columns = ["xau_usd", "fx"]
    return (aligned["xau_usd"] * aligned["fx"]).rename("XAU-USD (INR)")


if __name__ == "__main__":
    # Quick smoke test: run `python data.py` to sanity-check the pipeline
    # against your actual data file.
    df = build_return_dataset(real=True)
    print(f"Loaded {len(df)} months of real returns, {df.index.min():%b-%Y} to {df.index.max():%b-%Y}")
    print(df.head())
    print()
    print("Rows per regime:")
    print(df["regime"].value_counts())
    print()
    print("Months with a usable risk-free rate:", df["risk_free"].notna().sum(), "/", len(df))
