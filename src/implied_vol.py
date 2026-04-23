"""
implied_vol.py — Implied volatility computation and surface construction.

Implied volatility (IV) is the volatility that, when plugged into
Black-Scholes, produces the observed market price. It is the market's
consensus forecast of future volatility embedded in option prices.

Key findings this module demonstrates:
1. IV is NOT constant across strikes — the volatility smile/skew
2. IV is NOT constant across maturities — the volatility term structure
3. Together these form the volatility SURFACE

The volatility smile directly contradicts the BS assumption of constant
volatility. The market assigns higher probability to extreme moves
(fat tails) than BS implies — especially for equity index options
where downside crashes are more likely than upside jumps (skew).

Since we don't have real Vietnamese options market data, we:
1. Generate synthetic market prices using BS with a realistic smile
2. Add small noise to simulate bid-ask spread
3. Back out implied vols from these synthetic prices
4. Visualise the resulting surface

This is explicitly documented — the methodology is real even if
the inputs are synthetic.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes price — used internally for IV computation."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K * np.exp(-r * T), 0)
        else:
            return max(K * np.exp(-r * T) - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(market_price, S, K, T, r,
                option_type="call",
                lower=1e-6, upper=10.0):
    """
    Compute implied volatility using Brent's root-finding method.

    Finds σ* such that BS(S, K, T, r, σ*) = market_price.

    Brent's method is preferred over Newton-Raphson here because:
    - Guaranteed to converge (bracketed search)
    - Does not require derivatives (though vega is available)
    - Robust to poor initial guesses

    Parameters
    ----------
    market_price : observed option price in market
    S, K, T, r   : standard BS parameters
    option_type  : 'call' or 'put'
    lower, upper : search bounds for volatility

    Returns
    -------
    float : implied volatility, or NaN if no solution found
    """
    # Check if price is within no-arbitrage bounds.
    # Lower bound: discounted intrinsic value (forward intrinsic).
    # Upper bound: call ≤ S  /  put ≤ K·e^(-rT)
    if option_type == "call":
        intrinsic   = max(S - K * np.exp(-r * T), 0)
        upper_bound = S
    else:
        intrinsic   = max(K * np.exp(-r * T) - S, 0)
        upper_bound = K * np.exp(-r * T)

    if market_price <= intrinsic:
        return np.nan
    if market_price >= upper_bound:
        return np.nan

    # Objective: BS(σ) - market_price = 0
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, lower, upper, xtol=1e-8, maxiter=1000)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def generate_synthetic_smile(S, r, T,
                              moneyness=[0.80, 0.85, 0.90, 0.95,
                                         1.0, 1.05, 1.10, 1.15, 1.20],
                              atm_vol=0.20,
                              skew=-0.10,
                              smile_curve=0.05,
                              noise_pct=0.02,
                              seed=42):
    """
    Generate synthetic market prices with a realistic volatility smile.

    The smile is parametrised as:
        IV(K) = atm_vol + skew × (K/S - 1) + smile_curve × (K/S - 1)²

    Parameters:
    - atm_vol     : ATM implied vol (e.g. 0.20 = 20%)
    - skew        : slope of the smile (negative for equity = put skew)
    - smile_curve : curvature (positive = U-shape smile)
    - noise_pct   : random noise as % of price (simulates bid-ask)

    For equity indices, skew is typically negative because:
    - Downside crashes more likely than upside jumps
    - Investors pay premium for downside protection (puts)
    - This makes OTM put IVs higher than OTM call IVs

    Returns DataFrame with strikes, true IVs, prices, and noisy prices.
    """
    np.random.seed(seed)
    rows = []

    for m in moneyness:
        K = S * m

        # True IV at this strike (smile parametrisation)
        true_iv = atm_vol + skew * (m - 1) + smile_curve * (m - 1)**2

        # Ensure IV is positive
        true_iv = max(true_iv, 0.01)

        # BS price at true IV
        price = bs_price(S, K, T, r, true_iv, "call")

        # Add noise to simulate bid-ask spread
        noise        = np.random.uniform(-noise_pct, noise_pct) * price
        noisy_price  = max(price + noise, 0.001)

        rows.append({
            "moneyness"  : m,
            "K"          : round(K, 2),
            "true_iv"    : round(true_iv, 4),
            "bs_price"   : round(price, 4),
            "market_price": round(noisy_price, 4),
        })

    return pd.DataFrame(rows)


def compute_smile(S, r, T, atm_vol=0.20, skew=-0.10,
                  smile_curve=0.05, noise_pct=0.02,
                  moneyness=None, seed=42):
    """
    Generate synthetic market prices and back out implied vols.

    This is the full workflow:
    1. Generate prices with known smile parameters
    2. Add noise
    3. Back out IV from noisy prices using Brent's method
    4. Compare recovered IV vs true IV

    Returns DataFrame with true and recovered IVs.
    """
    if moneyness is None:
        moneyness = [0.80, 0.85, 0.90, 0.95, 1.0,
                     1.05, 1.10, 1.15, 1.20]

    # Generate synthetic market data
    df = generate_synthetic_smile(
        S, r, T, moneyness, atm_vol, skew,
        smile_curve, noise_pct, seed
    )

    # Back out implied vols from noisy market prices
    df["implied_vol"] = df.apply(
        lambda row: implied_vol(
            row["market_price"], S, row["K"], T, r, "call"
        ),
        axis=1
    )

    df["iv_error"] = (df["implied_vol"] - df["true_iv"]).round(4)

    return df


def build_vol_surface(S, r, atm_vol=0.20, skew=-0.10,
                      smile_curve=0.05, noise_pct=0.01,
                      maturities=[1/12, 3/12, 6/12, 1.0, 2.0],
                      moneyness=None):
    """
    Build volatility surface across strikes and maturities.

    In practice, the smile shape changes with maturity:
    - Short maturities: more pronounced smile (higher uncertainty short term)
    - Long maturities: flatter smile (mean reversion smooths extremes)

    We model this by making skew and curvature maturity-dependent:
        skew(T)  = skew / √T   (skew less pronounced for longer maturities)
        curve(T) = smile_curve × √T

    Returns:
    - surface_iv  : DataFrame of IVs (rows=moneyness, cols=maturities)
    - surface_price: DataFrame of prices
    """
    if moneyness is None:
        moneyness = [0.80, 0.85, 0.90, 0.95, 1.0,
                     1.05, 1.10, 1.15, 1.20]

    iv_surface    = {}
    price_surface = {}

    for T in maturities:
        # Maturity-adjusted smile parameters
        t_skew  = skew / np.sqrt(T * 12)   # less skew for longer maturities
        t_curve = smile_curve * np.sqrt(T * 12)

        smile_df = compute_smile(
            S, r, T,
            atm_vol     = atm_vol,
            skew        = t_skew,
            smile_curve = t_curve,
            noise_pct   = noise_pct,
            moneyness   = moneyness,
        )

        col = f"T={int(T*12)}m"
        iv_surface[col]    = smile_df["implied_vol"].values
        price_surface[col] = smile_df["market_price"].values

    index = [f"{int(m*100)}% ATM" for m in moneyness]

    return (
        pd.DataFrame(iv_surface, index=index),
        pd.DataFrame(price_surface, index=index),
    )


def garch_smile(S, r, T, garch_params,
                moneyness=None, n_paths=20000, seed=42,
                option_type="call"):
    """
    Endogenous volatility smile derived from GARCH Monte Carlo.

    The earlier `compute_smile` / `build_vol_surface` functions
    generate a smile synthetically from assumed skew / curvature
    parameters, then recover those same parameters by inverting BS
    on the noisy prices. That's methodologically sound as a
    demonstration but **circular** — the recovered smile is by
    construction equal (up to noise) to what you put in.

    This function is NOT circular. It:
      1. Prices options across strikes using the Duan GARCH MC.
      2. Inverts Black–Scholes on each MC price to get the BS-
         equivalent implied vol.
      3. Returns that curve.

    Because GARCH produces a non-log-normal terminal distribution
    (heavy tails from vol clustering), inverting the BS formula on
    those prices gives a genuine non-flat smile — one that is an
    output of the model rather than an input. This is the first
    time `implied_vol` in this project refers to something real.

    Parameters
    ----------
    garch_params : dict from garch_vol.fit_garch
    moneyness    : list of K/S ratios; default spans 0.80 → 1.20
    n_paths      : MC paths per strike

    Returns
    -------
    DataFrame with columns:
        moneyness, K, garch_mc_price, implied_vol
    """
    from monte_carlo import mc_garch

    if moneyness is None:
        moneyness = [0.80, 0.85, 0.90, 0.95, 1.0,
                     1.05, 1.10, 1.15, 1.20]

    rows = []
    for m in moneyness:
        K  = S * m
        mc = mc_garch(S, K, T, r, garch_params,
                      n_paths=n_paths, seed=seed,
                      option_type=option_type)
        iv = implied_vol(mc["price"], S, K, T, r, option_type)
        rows.append({
            "moneyness"     : m,
            "K"             : round(K, 2),
            "garch_mc_price": round(mc["price"], 4),
            "std_error"     : round(mc["std_error"], 4),
            "implied_vol"   : round(iv, 4) if np.isfinite(iv) else np.nan,
        })

    return pd.DataFrame(rows)


def term_structure(S, K, r, atm_vol=0.20,
                   maturities=[1/52, 1/12, 3/12,
                                6/12, 1.0, 2.0]):
    """
    Compute ATM implied vol term structure across maturities.

    The term structure shows how ATM IV changes with maturity.
    Common shapes:
    - Upward sloping: market expects more vol in future (calm now)
    - Downward sloping: market expects vol to mean-revert (volatile now)
    - Humped: near-term event risk then calm

    We model a simple mean-reverting term structure:
        IV(T) = long_run_vol + (atm_vol - long_run_vol) × e^(-κT)
    """
    long_run = atm_vol * 0.9  # long-run vol slightly below spot vol
    kappa    = 2.0            # mean reversion speed

    rows = []
    for T in maturities:
        iv   = long_run + (atm_vol - long_run) * np.exp(-kappa * T)
        price = bs_price(S, K, T, r, iv, "call")
        rows.append({
            "Maturity (months)": round(T * 12, 1),
            "Implied Vol (%)"  : round(iv * 100, 2),
            "Call Price"       : round(price, 4),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import get_model_inputs

    inputs = get_model_inputs(ticker="VCB", vol_window=252)
    S      = inputs["S"]
    r      = inputs["r"]
    sigma  = inputs["sigma"]

    # Use longer-term vol as ATM vol
    atm_vol = sigma

    print(f"\n{'='*60}")
    print(f"  Implied Volatility Analysis — {inputs['ticker']}")
    print(f"{'='*60}")
    print(f"  S={S:.1f}  r={r*100:.1f}%  ATM vol={atm_vol*100:.2f}%")

    # Single smile at 3 months
    print(f"\nVolatility Smile (T=3 months):")
    smile = compute_smile(
        S, r, T=0.25,
        atm_vol     = atm_vol,
        skew        = -0.10,
        smile_curve = 0.05,
        noise_pct   = 0.02,
    )
    print(smile[["moneyness", "K", "true_iv",
                  "implied_vol", "iv_error"]].to_string(index=False))

    # Vol surface
    print(f"\nImplied Volatility Surface:")
    iv_surf, price_surf = build_vol_surface(
        S, r,
        atm_vol     = atm_vol,
        skew        = -0.10,
        smile_curve = 0.05,
    )
    print((iv_surf * 100).round(2).to_string())

    # Term structure
    print(f"\nATM Vol Term Structure:")
    ts = term_structure(S, S, r, atm_vol)
    print(ts.to_string(index=False))

    # Save
    import os
    os.makedirs("data/processed", exist_ok=True)
    iv_surf.to_csv("data/processed/vol_surface.csv")
    print("\nSaved vol surface to data/processed/vol_surface.csv")