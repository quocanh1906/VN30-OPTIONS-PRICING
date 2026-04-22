"""
binomial.py — Cox-Ross-Rubinstein (CRR) binomial tree pricing.

Implements the binomial tree model for pricing both European
and American options. Unlike Black-Scholes which assumes
continuous trading, the binomial tree works in discrete time
steps — more realistic and more flexible.

Key advantages over Black-Scholes:
1. Handles American options (early exercise) naturally
2. Converges to Black-Scholes as N → infinity
3. Intuitive — can visualise the full price tree
4. Handles discrete dividends easily

CRR parametrisation:
    u = e^(σ√Δt)        up factor
    d = 1/u = e^(-σ√Δt) down factor
    p = (e^(rΔt) - d) / (u - d)  risk-neutral probability of up move

Parameters:
    S : spot price
    K : strike price
    T : time to expiry in years
    r : risk-free rate per annum
    sigma : volatility per annum
    N : number of time steps (more steps = more accurate)
"""

import numpy as np
import pandas as pd


def _crr_params(T, r, sigma, N):
    """
    Compute CRR binomial tree parameters.

    Returns dt, u, d, p (risk-neutral up probability).
    """
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1 / u
    p  = (np.exp(r * dt) - d) / (u - d)
    return dt, u, d, p


def european_price(S, K, T, r, sigma, N=200, option_type="call"):
    """
    Price a European option using the CRR binomial tree.

    European options can only be exercised at expiry.
    With N=200 steps, price converges very close to Black-Scholes.

    Algorithm:
    1. Build stock price tree (forward pass)
    2. Compute payoffs at expiry (terminal nodes)
    3. Discount backwards through tree (backward induction)

    Parameters
    ----------
    N : number of time steps (default 200 — good accuracy)
    """
    dt, u, d, p = _crr_params(T, r, sigma, N)
    discount    = np.exp(-r * dt)

    # Terminal stock prices at expiry
    # At node (N, j): price = S × u^j × d^(N-j)
    j           = np.arange(0, N + 1)
    ST          = S * (u ** j) * (d ** (N - j))

    # Terminal payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    # Backward induction — discount one step at a time
    for _ in range(N):
        payoffs = discount * (p * payoffs[1:] + (1 - p) * payoffs[:-1])

    return payoffs[0]


def american_price(S, K, T, r, sigma, N=200, option_type="put"):
    """
    Price an American option using the CRR binomial tree.

    American options can be exercised at ANY time before expiry.
    At each node, holder compares:
        - Intrinsic value (exercise now)
        - Continuation value (hold and wait)
    Takes the maximum — rational early exercise.

    American calls on non-dividend paying stocks are never exercised early
    (same price as European). American puts can be exercised early when
    deeply in the money.

    Algorithm:
    1. Build full stock price tree
    2. Compute payoffs at terminal nodes
    3. Backward induction with early exercise check at each node
    """
    dt, u, d, p = _crr_params(T, r, sigma, N)
    discount    = np.exp(-r * dt)

    # Build full stock price tree
    # stock[i, j] = price at time step i, having gone up j times
    stock = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock[i, j] = S * (u ** j) * (d ** (i - j))

    # Terminal payoffs
    if option_type == "call":
        option = np.maximum(stock[N, :N+1] - K, 0)
    else:
        option = np.maximum(K - stock[N, :N+1], 0)

    # Backward induction with early exercise
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Continuation value
            continuation = discount * (
                p * option[j + 1] + (1 - p) * option[j]
            )
            # Intrinsic value (exercise now)
            if option_type == "call":
                intrinsic = max(stock[i, j] - K, 0)
            else:
                intrinsic = max(K - stock[i, j], 0)

            # American: take maximum of exercise vs hold
            option[j] = max(continuation, intrinsic)

    return option[0]


def early_exercise_premium(S, K, T, r, sigma, N=200):
    """
    Compute the early exercise premium for an American put.

    Early exercise premium = American put - European put

    This premium is always >= 0 since American >= European.
    It increases when:
    - Option is deep in the money (high intrinsic value)
    - Interest rates are high (benefit of receiving cash early)
    - Time to expiry is long

    Returns dict with both prices and premium.
    """
    american = american_price(S, K, T, r, sigma, N, "put")
    european = european_price(S, K, T, r, sigma, N, "put")
    premium  = american - european

    return {
        "American Put"          : round(american, 4),
        "European Put"          : round(european, 4),
        "Early Exercise Premium": round(premium, 4),
        "Premium as % of Euro"  : round(premium / european * 100, 2)
                                  if european > 0 else 0,
    }


def convergence_analysis(S, K, T, r, sigma,
                          steps=[5, 10, 20, 50, 100, 200, 500],
                          option_type="call"):
    """
    Show how binomial price converges to Black-Scholes as N increases.

    Demonstrates that CRR binomial tree is a discretisation of the
    continuous-time Black-Scholes model — same answer, different method.

    Returns DataFrame with N, binomial price, BS price, and error.
    """
    from black_scholes import call_price, put_price

    bs_price = call_price(S, K, T, r, sigma) if option_type == "call" \
               else put_price(S, K, T, r, sigma)

    results = []
    for N in steps:
        bin_price = european_price(S, K, T, r, sigma, N, option_type)
        error     = bin_price - bs_price
        results.append({
            "N steps"       : N,
            "Binomial Price": round(bin_price, 6),
            "BS Price"      : round(bs_price, 6),
            "Error"         : round(error, 6),
            "Error (%)"     : round(abs(error / bs_price) * 100, 4)
                              if bs_price > 0 else 0,
        })

    return pd.DataFrame(results)


def price_tree_small(S, K, T, r, sigma, N=5, option_type="call"):
    """
    Build and display a small binomial tree for visualisation.

    Only useful for small N (e.g. N=5) to inspect the tree structure.
    Shows stock prices and option values at each node.

    Returns stock price tree and option value tree as DataFrames.
    """
    dt, u, d, p = _crr_params(T, r, sigma, N)
    discount    = np.exp(-r * dt)

    # Build stock tree
    stock = np.full((N + 1, N + 1), np.nan)
    for i in range(N + 1):
        for j in range(i + 1):
            stock[i, j] = round(S * (u ** j) * (d ** (i - j)), 2)

    # Terminal payoffs
    option = np.full((N + 1, N + 1), np.nan)
    for j in range(N + 1):
        if option_type == "call":
            option[N, j] = max(stock[N, j] - K, 0)
        else:
            option[N, j] = max(K - stock[N, j], 0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option[i, j] = round(
                discount * (p * option[i + 1, j + 1] +
                            (1 - p) * option[i + 1, j]), 4
            )

    stock_df  = pd.DataFrame(stock).T
    option_df = pd.DataFrame(option).T

    return stock_df, option_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import get_model_inputs
    from black_scholes import call_price, put_price

    inputs = get_model_inputs(ticker="VCB", vol_window=30)
    S      = inputs["S"]
    r      = inputs["r"]
    sigma  = inputs["sigma"]
    K      = S        # ATM
    T      = 0.25     # 3 months

    print(f"\n{'='*55}")
    print(f"  Binomial Tree: ATM Option on {inputs['ticker']}")
    print(f"{'='*55}")
    print(f"  S={S:.1f}  K={K:.1f}  T={T}y  r={r*100:.1f}%  σ={sigma*100:.2f}%")

    # European pricing
    bin_call = european_price(S, K, T, r, sigma, N=200, option_type="call")
    bin_put  = european_price(S, K, T, r, sigma, N=200, option_type="put")
    bs_call  = call_price(S, K, T, r, sigma)
    bs_put   = put_price(S, K, T, r, sigma)

    print(f"\nEuropean Prices (N=200 steps):")
    print(f"  {'':10} {'Binomial':>12} {'Black-Scholes':>15} {'Diff':>10}")
    print(f"  {'Call':10} {bin_call:>12.4f} {bs_call:>15.4f} "
          f"{bin_call-bs_call:>+10.6f}")
    print(f"  {'Put':10} {bin_put:>12.4f} {bs_put:>15.4f} "
          f"{bin_put-bs_put:>+10.6f}")

    # American put early exercise premium
    print(f"\nEarly Exercise Premium (American vs European Put):")
    eep = early_exercise_premium(S, K, T, r, sigma)
    for k, v in eep.items():
        print(f"  {k:<30} {v}")

    # Convergence analysis
    print(f"\nConvergence to Black-Scholes (Call):")
    conv = convergence_analysis(S, K, T, r, sigma,
                                steps=[5, 10, 20, 50, 100, 200, 500])
    print(conv.to_string(index=False))

    # Small tree visualisation
    print(f"\nSmall Tree (N=5 steps) — Stock Prices:")
    stock_tree, option_tree = price_tree_small(S, K, T, r, sigma, N=5)
    print(stock_tree.round(2).to_string())
    print(f"\nSmall Tree (N=5 steps) — Call Option Values:")
    print(option_tree.round(4).to_string())