"""
black_scholes.py — Analytical Black-Scholes pricing and Greeks.

Implements the Black-Scholes (1973) closed-form solution for
European call and put options, plus all five Greeks.

Assumptions:
- Underlying follows Geometric Brownian Motion
- Constant volatility σ over option lifetime
- Constant risk-free rate r
- No dividends
- No transaction costs
- Continuous trading possible

These assumptions are violated in practice — particularly constant vol
(the volatility smile shows this directly). Nevertheless BS remains
the industry standard quoting convention for options.

Parameters used throughout:
    S : spot price
    K : strike price
    T : time to expiry in years (e.g. 0.25 = 3 months)
    r : risk-free rate per annum (e.g. 0.045 = 4.5%)
    sigma : volatility per annum (e.g. 0.25 = 25%)
"""

import numpy as np
from scipy.stats import norm
import pandas as pd


# ── Core d1, d2 ─────────────────────────────────────────────────────────────────

def _d1(S, K, T, r, sigma):
    """
    Compute d1 — standardised distance to strike adjusted for drift.
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    """
    Compute d2 = d1 - σ√T
    N(d2) = risk-neutral probability option expires in the money.
    """
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


# ── Option Prices ───────────────────────────────────────────────────────────────

def call_price(S, K, T, r, sigma):
    """
    Black-Scholes European call price.

    Call = S × N(d1) - K × e^(-rT) × N(d2)

    Intuition:
    - S × N(d1) : expected value of receiving the stock if exercised
    - K × e^(-rT) × N(d2) : present value of paying the strike

    Returns 0 if T <= 0 (expired option).
    """
    if T <= 0:
        return max(S - K, 0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma):
    """
    Black-Scholes European put price.

    Put = K × e^(-rT) × N(-d2) - S × N(-d1)

    Derived from put-call parity:
    Put = Call - S + K × e^(-rT)

    Returns 0 if T <= 0 (expired option).
    """
    if T <= 0:
        return max(K - S, 0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def put_call_parity_check(S, K, T, r, sigma):
    """
    Verify put-call parity: C - P = S - K × e^(-rT)

    This is a model-free relationship that must hold to prevent arbitrage.
    Any deviation indicates a pricing error.

    Returns difference — should be ~0.
    """
    C   = call_price(S, K, T, r, sigma)
    P   = put_price(S, K, T, r, sigma)
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    return {
        "Call"           : round(C, 4),
        "Put"            : round(P, 4),
        "C - P"          : round(lhs, 4),
        "S - Ke^(-rT)"   : round(rhs, 4),
        "Parity error"   : round(lhs - rhs, 8),
        "Parity holds"   : abs(lhs - rhs) < 1e-6,
    }


# ── Greeks ──────────────────────────────────────────────────────────────────────

def delta(S, K, T, r, sigma, option_type="call"):
    """
    Delta — sensitivity of option price to spot price movement.

    Call delta = N(d1)  ∈ [0, 1]
    Put delta  = N(d1) - 1 ∈ [-1, 0]

    Interpretation: a call with delta=0.5 gains 0.5 VND
    for every 1 VND increase in the underlying.

    Delta is also the hedge ratio — you need delta shares
    of stock to hedge one option.
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """
    Gamma — rate of change of Delta with respect to spot price.
    Same for calls and puts (by put-call parity).

    Γ = N'(d1) / (S × σ × √T)

    Interpretation: if spot moves 1 VND, delta changes by gamma.
    High gamma means delta changes rapidly → more frequent rehedging needed.
    Gamma is highest for ATM options near expiry.
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Vega — sensitivity of option price to volatility.
    Same for calls and puts.

    ν = S × N'(d1) × √T

    Interpretation: vega=500 means option gains 500 VND
    if volatility increases by 1% (0.01).
    Usually quoted per 1% change in vol.

    Note: vega is not a Greek letter — it's a made-up term in finance.
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call"):
    """
    Theta — time decay of option price per day.

    Θ_call = -[S × N'(d1) × σ / (2√T)] - r × K × e^(-rT) × N(d2)
    Θ_put  = -[S × N'(d1) × σ / (2√T)] + r × K × e^(-rT) × N(-d2)

    Interpretation: theta=-50 means option loses 50 VND per day
    due to time decay, all else equal.
    Options are wasting assets — buyers lose theta, sellers earn it.

    Returned as per-day value (divided by 365).
    """
    if T <= 0:
        return 0.0
    d1  = _d1(S, K, T, r, sigma)
    d2  = _d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)

    return (term1 + term2) / 365  # per calendar day


def rho(S, K, T, r, sigma, option_type="call"):
    """
    Rho — sensitivity of option price to interest rate.

    ρ_call = K × T × e^(-rT) × N(d2)
    ρ_put  = -K × T × e^(-rT) × N(-d2)

    Interpretation: rho=200 means option gains 200 VND
    if risk-free rate increases by 1% (0.01).
    Less important than other Greeks for short-dated options.
    """
    if T <= 0:
        return 0.0
    d2 = _d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def all_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute all Greeks for one option in a single call.

    Returns dict with Delta, Gamma, Vega, Theta, Rho.
    """
    return {
        "Delta": round(delta(S, K, T, r, sigma, option_type), 6),
        "Gamma": round(gamma(S, K, T, r, sigma), 8),
        "Vega" : round(vega(S, K, T, r, sigma), 4),
        "Theta": round(theta(S, K, T, r, sigma, option_type), 4),
        "Rho"  : round(rho(S, K, T, r, sigma, option_type), 4),
    }


# ── Price surface ───────────────────────────────────────────────────────────────

def price_surface(S, r, sigma, option_type="call",
                  maturities=[1/12, 3/12, 6/12, 1.0],
                  moneyness=[0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20]):
    """
    Compute option prices across a grid of strikes and maturities.

    Parameters
    ----------
    S           : spot price
    r           : risk-free rate
    sigma       : volatility
    option_type : 'call' or 'put'
    maturities  : list of T in years
    moneyness   : list of K/S ratios

    Returns
    -------
    DataFrame with strikes as rows, maturities as columns
    """
    pricer = call_price if option_type == "call" else put_price

    strikes = [round(S * m) for m in moneyness]
    data    = {}

    for T in maturities:
        col_label = f"T={int(T*12)}m"
        data[col_label] = [
            round(pricer(S, K, T, r, sigma), 2)
            for K in strikes
        ]

    df = pd.DataFrame(data, index=[f"K={int(k)}" for k in strikes])
    df.index.name = "Strike"
    return df


def greeks_surface(S, r, sigma, option_type="call",
                   maturities=[1/12, 3/12, 6/12, 1.0],
                   moneyness=[0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20]):
    """
    Compute Greeks across a grid of strikes and maturities.

    Returns dict of {greek_name: DataFrame}.
    """
    strikes  = [round(S * m) for m in moneyness]
    greek_fns = {
        "Delta": lambda K, T: delta(S, K, T, r, sigma, option_type),
        "Gamma": lambda K, T: gamma(S, K, T, r, sigma),
        "Vega" : lambda K, T: vega(S, K, T, r, sigma),
        "Theta": lambda K, T: theta(S, K, T, r, sigma, option_type),
    }

    surfaces = {}
    for greek_name, fn in greek_fns.items():
        data = {}
        for T in maturities:
            col = f"T={int(T*12)}m"
            data[col] = [round(fn(K, T), 6) for K in strikes]
        surfaces[greek_name] = pd.DataFrame(
            data, index=[f"K={int(k)}" for k in strikes]
        )
        surfaces[greek_name].index.name = "Strike"

    return surfaces


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import get_model_inputs

    # Load real VCB inputs
    inputs = get_model_inputs(ticker="VCB", vol_window=30)
    S     = inputs["S"]
    r     = inputs["r"]
    sigma = inputs["sigma"]

    # ATM option example
    K = S        # at-the-money
    T = 0.25     # 3 months

    print(f"\n{'='*55}")
    print(f"  Black-Scholes: ATM Option on {inputs['ticker']}")
    print(f"{'='*55}")
    print(f"  S     = {S:,.0f} VND")
    print(f"  K     = {K:,.0f} VND (ATM)")
    print(f"  T     = {T} years (3 months)")
    print(f"  r     = {r*100:.1f}%")
    print(f"  sigma = {sigma*100:.2f}%")

    C = call_price(S, K, T, r, sigma)
    P = put_price(S, K, T, r, sigma)

    print(f"\n  Call price = {C:,.2f} VND")
    print(f"  Put  price = {P:,.2f} VND")

    # Put-call parity check
    print(f"\nPut-Call Parity Check:")
    pcp = put_call_parity_check(S, K, T, r, sigma)
    for k, v in pcp.items():
        print(f"  {k:<20} {v}")

    # Greeks
    print(f"\nCall Greeks (ATM, 3M):")
    greeks = all_greeks(S, K, T, r, sigma, "call")
    for g, val in greeks.items():
        print(f"  {g:<10} {val}")

    print(f"\nPut Greeks (ATM, 3M):")
    greeks_p = all_greeks(S, K, T, r, sigma, "put")
    for g, val in greeks_p.items():
        print(f"  {g:<10} {val}")

    # Price surface
    print(f"\nCall Price Surface:")
    surf = price_surface(S, r, sigma, "call")
    print(surf.to_string())

    # Greeks surface — Delta
    print(f"\nDelta Surface (Call):")
    gsurfs = greeks_surface(S, r, sigma, "call")
    print(gsurfs["Delta"].to_string())