"""
garch_vol.py — GARCH(1,1) volatility estimation and forecasting.

Motivation
----------
Vietnam does not have a liquid listed-options market, so there is no
implied-volatility input to borrow from the market. Our only source of
information about volatility is the historical return series itself.

The naive approach — annualising the standard deviation of recent log
returns over a fixed window — treats volatility as constant. Empirical
returns violate this in two well-documented ways:

  1. Volatility clustering: large moves follow large moves.
  2. Mean reversion: volatility eventually returns to a long-run level.

GARCH(1,1) (Bollerslev 1986) models both:

    σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}

where ε_{t-1} is last period's mean-zero return shock. This is a single
equation with three free parameters that captures both the burst of
vol after shocks (α term) and the slow mean-reversion back to the
long-run unconditional vol (β term).

Two outputs relevant for option pricing
---------------------------------------
Level 1 (scalar σ for Black–Scholes / binomial):
  The average conditional variance over the option's life, annualised.
  Collapses the time-varying GARCH process to a single number that
  still carries the vol-clustering information via the current σ_t.

Level 2 (full path for Duan Monte Carlo):
  The fitted parameters (ω, α, β) plus the current variance σ²_t. These
  let a Monte Carlo engine evolve the variance day-by-day under the
  risk-neutral measure, producing an endogenous volatility smile.

Library
-------
Uses `arch` (Kevin Sheppard) — the de facto Python implementation. We
pass returns scaled by 100 so fit is numerically stable; variance
returned is in percent-squared and must be scaled back to fractional
units (÷ 10,000) before use downstream.
"""

import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(returns, p=1, q=1):
    """
    Fit a GARCH(p, q) model with zero-mean assumption.

    Zero-mean is a standard simplification for daily equity returns —
    the mean is tiny relative to daily vol, and estimating it
    introduces noise. We care about variance dynamics, not drift.

    Parameters
    ----------
    returns : Series of log returns (fractional units, not percent)
    p, q    : GARCH orders (defaults (1,1))

    Returns
    -------
    dict with:
        omega, alpha, beta   : fitted GARCH parameters (per day,
                               in fractional-return units)
        persistence          : α + β, must be < 1 for stationarity
        long_run_variance    : ω / (1 − α − β) (daily)
        long_run_vol_annual  : √(long_run_variance · 252)
        last_variance        : σ²_t at end of sample (daily)
        last_vol_annual      : √(last_variance · 252)
        result               : full arch fit object (for diagnostics)
    """
    # arch expects returns in percent for numerical stability
    r_pct = returns.dropna() * 100.0

    am  = arch_model(r_pct, mean="Zero", vol="Garch", p=p, q=q,
                     dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)

    # Rescale back to fractional units
    # Variance in %² → fractional² by ÷10⁴
    omega = res.params["omega"]         / 10_000
    alpha = res.params["alpha[1]"]
    beta  = res.params["beta[1]"]

    persistence = alpha + beta
    long_run_variance = omega / (1 - persistence) \
                        if persistence < 1 else np.nan

    last_variance = res.conditional_volatility.iloc[-1]**2 / 10_000

    return {
        "omega"               : omega,
        "alpha"               : alpha,
        "beta"                : beta,
        "persistence"         : persistence,
        "long_run_variance"   : long_run_variance,
        "long_run_vol_annual" : np.sqrt(long_run_variance * 252)
                                if persistence < 1 else np.nan,
        "last_variance"       : last_variance,
        "last_vol_annual"     : np.sqrt(last_variance * 252),
        "result"              : res,
    }


def forecast_variance_path(garch_params, n_days):
    """
    Project daily conditional variance forward under the fitted GARCH.

    Uses the analytical multi-step GARCH forecast (not simulation):

        E[σ²_{t+h} | F_t] = ω + (α + β) · E[σ²_{t+h-1} | F_t]

    This decays geometrically toward the long-run variance at rate
    (α + β) per day. The decay is the mean reversion.

    Returns
    -------
    array of length n_days, daily variance forecasts σ²_{t+1..t+n_days}
    """
    omega    = garch_params["omega"]
    alpha    = garch_params["alpha"]
    beta     = garch_params["beta"]
    var_path = np.zeros(n_days)
    var_path[0] = omega + (alpha + beta) * garch_params["last_variance"]
    for h in range(1, n_days):
        var_path[h] = omega + (alpha + beta) * var_path[h - 1]
    return var_path


def garch_sigma_for_horizon(garch_params, T, trading_days=252):
    """
    Level 1 input for BS / binomial: a single σ over horizon T.

    Method: forecast the daily variance path for `ceil(T · 252)` days,
    average those daily variances, then annualise. This is the "right"
    scalar to plug into a constant-vol model — it captures the current
    vol regime through the starting σ²_t, and the mean reversion
    through the forecast path, without pretending that vol is flat.

    Parameters
    ----------
    garch_params : output of fit_garch
    T            : horizon in years
    trading_days : 252 (convention)

    Returns
    -------
    scalar σ, annualised
    """
    n_days = max(int(np.ceil(T * trading_days)), 1)
    var_path = forecast_variance_path(garch_params, n_days)
    return float(np.sqrt(var_path.mean() * trading_days))


def in_sample_conditional_vol(garch_params):
    """
    Daily in-sample conditional volatilities from the fit, annualised.

    Useful for plotting GARCH σ_t against rolling historical vol to
    show how GARCH reacts to vol clusters more sharply than a flat
    N-day rolling window.

    Returns
    -------
    Series of annualised σ_t, indexed same as input returns
    """
    res = garch_params["result"]
    # conditional_volatility is in percent (since we fit on %-returns)
    return pd.Series(
        res.conditional_volatility.values / 100.0 * np.sqrt(252),
        index=res.conditional_volatility.index,
        name="garch_cond_vol",
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import load_prices

    prices      = load_prices("VCB")
    log_returns = np.log(prices / prices.shift(1)).dropna()

    print("Fitting GARCH(1,1) on VCB daily log returns...")
    g = fit_garch(log_returns)

    print(f"\n  ω (omega)            : {g['omega']:.2e}")
    print(f"  α (alpha)            : {g['alpha']:.4f}")
    print(f"  β (beta)             : {g['beta']:.4f}")
    print(f"  persistence (α+β)    : {g['persistence']:.4f}")
    print(f"  long-run vol (annual): {g['long_run_vol_annual']*100:.2f}%")
    print(f"  last vol (annual)    : {g['last_vol_annual']*100:.2f}%")

    print("\nLevel-1 σ for BS/binomial at different horizons:")
    for T in [1/12, 3/12, 6/12, 1.0]:
        s = garch_sigma_for_horizon(g, T)
        print(f"  T={int(T*12)}m  →  σ = {s*100:.2f}%")
