"""
data.py — Data handler for VN30 Options Pricing Engine.

Responsibilities:
- Load VN30 daily price data from existing processed CSV
  (reuses data already downloaded in VN30-Momentum project)
- Compute historical volatility at multiple lookback windows
- Compute basic statistics used as BS model inputs
- Provide a clean interface for other modules to get
  spot price, volatility, and risk-free rate

No new data download needed — uses VN30 prices already on disk.
Risk-free rate approximated from Vietnamese government bond yields.
"""

import pandas as pd
import numpy as np
import os


# ── Risk-free rate ──────────────────────────────────────────────────────────────
# Vietnamese 1-year government bond yield (approximate, 2024)
# Source: State Bank of Vietnam
RISK_FREE_RATE = 0.045  # 4.5% per annum

# Default underlying — use VN30 ETF as proxy for index options
DEFAULT_TICKER = "VCB"  # most liquid VN30 stock as example

# HOSE quotes prices in thousands of VND (e.g. 60.56 → 60,560 VND)
PRICE_UNIT = 1000


def load_prices(ticker=DEFAULT_TICKER,
                data_path=None):
    """
    Load daily close prices for a VN30 stock.

    Tries to load from VN30-Momentum project first (avoids re-downloading).
    Falls back to VN30-MA-Crossover data if available.

    Parameters
    ----------
    ticker    : stock ticker e.g. 'VCB', 'HPG', 'VNM'
    data_path : explicit path to prices CSV (optional)

    Returns
    -------
    pd.Series of daily close prices
    """
    # Search paths in priority order
    search_paths = [
        data_path,
        os.path.expanduser(
            "~/Github Projects/Momentum-Vietnam/data/processed/prices_weekly.csv"
        ),
        os.path.expanduser(
            "~/Github Projects/VN30-MA-Crossover/data/processed/closes_daily.csv"
        ),
        os.path.expanduser(
            "~/Github Projects/VN30-Momentum/data/processed/prices.csv"
        ),
    ]

    for path in search_paths:
        if path and os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if ticker in df.columns:
                prices = df[ticker].dropna() * PRICE_UNIT
                print(f"✓ Loaded {ticker}: {len(prices)} observations "
                      f"({prices.index[0].date()} to "
                      f"{prices.index[-1].date()}) from {path}")
                return prices

    raise FileNotFoundError(
        f"Could not find price data for {ticker}. "
        f"Run data.py in VN30-MA-Crossover first."
    )


def compute_historical_vol(prices, windows=[30, 60, 252]):
    """
    Compute annualised historical volatility at multiple lookback windows.

    Uses log returns for volatility estimation — standard in options pricing.
    Log returns are more appropriate than simple returns because:
    - They are additive over time
    - GBM (underlying BS model assumption) implies log-normal prices

    Parameters
    ----------
    prices  : Series of daily close prices
    windows : list of lookback days e.g. [30, 60, 252]

    Returns
    -------
    DataFrame with one column per window, index = dates
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()

    vols = {}
    for w in windows:
        vol = log_returns.rolling(w).std() * np.sqrt(252)
        vols[f"vol_{w}d"] = vol

    return pd.DataFrame(vols).dropna()


def get_model_inputs(ticker=DEFAULT_TICKER,
                     vol_window=30,
                     T=0.25,
                     data_path=None,
                     fit_garch_model=True):
    """
    Get all inputs needed for options pricing models.

    Returns two σ estimates side by side so downstream code can
    compare them:

    - `sigma`        : historical σ over the last `vol_window` days
                       (backward-looking, flat, reflects one recent
                       regime)
    - `sigma_garch`  : GARCH(1,1) Level-1 forecast — average
                       conditional variance over [0, T] annualised
                       (forward-looking, regime-aware, respects
                       mean reversion)

    The full `garch_params` dict is also returned so the Duan MC
    (Level 2) pricer can evolve the variance process day-by-day.

    Parameters
    ----------
    ticker          : stock ticker
    vol_window      : lookback for historical-vol baseline (days)
    T               : horizon for GARCH scalar forecast (years)
    data_path       : optional explicit path to price data
    fit_garch_model : skip GARCH fit if False (faster for tests)
    """
    prices      = load_prices(ticker, data_path)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    vol_df      = compute_historical_vol(prices, windows=[30, 60, 252])

    S     = prices.iloc[-1]
    sigma = log_returns.tail(vol_window).std() * np.sqrt(252)
    r     = RISK_FREE_RATE

    garch_params = None
    sigma_garch  = None
    if fit_garch_model:
        from garch_vol import fit_garch, garch_sigma_for_horizon
        garch_params = fit_garch(log_returns)
        sigma_garch  = garch_sigma_for_horizon(garch_params, T)

    print(f"\nModel inputs for {ticker}:")
    print(f"  Spot price (S)     : {S:,.0f} VND")
    print(f"  Risk-free rate (r) : {r*100:.1f}%")
    print(f"  Hist. vol ({vol_window}d)    : {sigma*100:.2f}%")
    print(f"  30d vol (baseline) : {vol_df['vol_30d'].iloc[-1]*100:.2f}%")
    print(f"  60d vol            : {vol_df['vol_60d'].iloc[-1]*100:.2f}%")
    print(f"  252d vol           : {vol_df['vol_252d'].iloc[-1]*100:.2f}%")
    if garch_params is not None:
        print(f"  GARCH σ_t (annual) : "
              f"{garch_params['last_vol_annual']*100:.2f}%")
        print(f"  GARCH long-run     : "
              f"{garch_params['long_run_vol_annual']*100:.2f}%")
        print(f"  GARCH forecast σ   : {sigma_garch*100:.2f}%  "
              f"(avg over [0, {T:.2f}y])")

    return {
        "ticker"       : ticker,
        "S"            : S,
        "r"            : r,
        "sigma"        : sigma,
        "sigma_garch"  : sigma_garch,
        "garch_params" : garch_params,
        "prices"       : prices,
        "returns"      : log_returns,
        "vol_series"   : vol_df,
    }


def get_option_grid(S, sigma, maturities=[1/12, 3/12, 6/12, 1.0],
                    moneyness=[0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20]):
    """
    Generate a grid of option strikes and maturities.

    Strikes computed as moneyness × spot price.
    Moneyness = K/S (1.0 = at-the-money)

    Parameters
    ----------
    S          : spot price
    sigma      : volatility (used for reference only)
    maturities : list of time to expiry in years
    moneyness  : list of K/S ratios

    Returns
    -------
    DataFrame with columns: K, T, moneyness
    """
    rows = []
    for T in maturities:
        for m in moneyness:
            K = S * m
            rows.append({
                "T"         : T,
                "K"         : round(K, 0),
                "moneyness" : m,
                "label"     : f"T={int(T*12)}m K={m:.0%}",
            })

    df = pd.DataFrame(rows)
    print(f"\nOption grid: {len(df)} contracts")
    print(f"  Maturities : {[f'{int(t*12)}m' for t in maturities]}")
    print(f"  Strikes    : {[f'{m:.0%}' for m in moneyness]} of spot")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Test loading and computing inputs
    inputs = get_model_inputs(ticker="VCB", vol_window=30)

    # Show vol term structure
    print("\nVolatility term structure (last observation):")
    vol_df = inputs["vol_series"]
    for col in vol_df.columns:
        print(f"  {col}: {vol_df[col].iloc[-1]*100:.2f}%")

    # Show option grid
    grid = get_option_grid(
        S         = inputs["S"],
        sigma     = inputs["sigma"],
        maturities= [1/12, 3/12, 6/12, 1.0],
        moneyness = [0.90, 0.95, 1.0, 1.05, 1.10],
    )
    print(f"\n{grid.to_string(index=False)}")

    # Save vol series
    os.makedirs("data/processed", exist_ok=True)
    vol_df.to_csv("data/processed/volatility.csv")
    print("\nSaved volatility series to data/processed/volatility.csv")