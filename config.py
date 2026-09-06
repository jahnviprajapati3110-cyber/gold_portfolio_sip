"""
config.py
---------
Every constant that the rest of the project depends on lives here, so that
changing an asset proxy, a regime boundary, or the allocation floor never
requires touching data.py, optimizer.py, or app.py.
"""

from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "Portfolio_RawData_Input.xlsx"

# ---------------------------------------------------------------------------
# Primary assets used in the main analysis (Phase 0-4)
# Keys are the internal names used everywhere in the code; values are the
# exact sheet names in the raw Excel workbook.
# ---------------------------------------------------------------------------
PRIMARY_ASSETS = {
    "equity": "Nifty TRI",
    "bond": "Nifty 10Yr G-Sec Index",
    "gold": "Nippon Gold BeES",
}
ASSET_ORDER = ["equity", "bond", "gold"]  # fixed order used for weight vectors everywhere

RISK_FREE_SHEET = "91-Day T-Bill"
CPI_SHEET = "CPI Monthly"

# ---------------------------------------------------------------------------
# Robustness proxies (Phase 5) -- alternates for gold and bond, swapped in
# one at a time in place of the primary asset above.
# ---------------------------------------------------------------------------
GOLD_PROXIES = {
    "Nippon Gold BeES (primary)": "Nippon Gold BeES",
    "MCX Gold": "MCX Gold",
    "XAU-USD (needs FX conversion)": "XAU-USD",
    "HDFC Gold ETF": "HDFC Gold ETF",
    "SBI Gold ETF": "SBI Gold ETF",
}
BOND_PROXIES = {
    "Nifty 10Yr G-Sec Index (primary)": "Nifty 10Yr G-Sec Index",
    "ICICI G-Sec ETF (short history)": "ICICI G-Sec ETF",
    "UTI G-Sec ETF (short history)": "UTI G-Sec ETF",
}
FX_SHEET = "USD-INR FX"  # needed to convert XAU-USD into INR terms

# ---------------------------------------------------------------------------
# Regime boundaries (inclusive month-end dates). Anything after the last
# boundary falls into "Post-Covid" automatically -- see data.phase_of().
# ---------------------------------------------------------------------------
REGIME_BOUNDARIES = {
    "Pre-Covid": datetime(2020, 2, 29),
    "Covid": datetime(2023, 5, 31),
    # "Post-Covid": everything after Covid's boundary
}
REGIME_ORDER = ["Pre-Covid", "Covid", "Post-Covid", "Full Sample"]

# ---------------------------------------------------------------------------
# Optimization defaults
# ---------------------------------------------------------------------------
DEFAULT_FLOOR = 0.25          # minimum weight per asset (Joshipura & Joshipura)
N_ASSETS = 3
DEFAULT_FRONTIER_POINTS = 20
DEFAULT_ROLLING_WINDOW_MONTHS = 36   # 3-year trailing window for Phase 4

# CPI base-year rebasing: MOSPI switched base year here. Month-over-month
# CPI index ratios are NOT valid across this boundary -- see data.py.
CPI_REBASE_BOUNDARY = datetime(2026, 1, 1)  # first month on the new (2024=100) base
