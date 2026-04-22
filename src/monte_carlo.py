"""
monte_carlo.py — Monte Carlo option pricing with variance reduction.

Simulates thousands of stock price paths under risk-neutral measure
(Geometric Brownian Motion) and averages discounted payoffs.

Three pricing approaches implemented:
1. Naive Monte Carlo — baseline, no variance reduction
2. Antithetic variates — pairs each path with its mirror image,
   reduces variance by ~50% at no extra simulation cost
3. Control variates — uses Black-Scholes as a known control,
   exploits correlation between MC estimate and BS price
   to dramatically reduce variance

Why Monte Carlo?
- Most flexible pricing method — works for any payoff structure
- Naturally handles path-dependent options (Asian, barrier, lookback)
- Trivially parallelisable
- Provides confidence intervals on price estimates
- Black-Scholes and binomial cannot handle exotic payoffs easily

Key limitation:
- Slow convergence: error decreases as 1/√N
  (to halve error, need 4x more simulations)
- Cannot efficiently price American options
  (requires Longstaff-Schwartz LSM — not implemented here)

GBM path simulation:
    S_t = S_0 × exp((r - σ²/2)×t + σ×√t×Z)
    where Z ~ N(0,1) is standard normal
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def simulate_gbm(S, r, sigma, T, n_paths=10000, n_steps=252, seed=42):
    """
    Simulate stock price paths under Geometric Brownian Motion.

    Uses the exact GBM solution (not Euler discretisation) to avoid
    discretisation error. Each step compounds the previous price.

    Parameters
    ----------
    S        : initial spot price
    r        : risk-free rate (risk-neutral drift)
    sigma    : volatility
    T        : time to expiry in years
    n_paths  : number of simulated paths
    n_steps  : number of time steps per path (252 = daily)
    seed     : random seed for reproducibility

    Returns
    -------
    paths : array of shape (n_steps+1, n_paths)
            paths[0] = S for all paths (starting price)
            paths[-1] = terminal prices
    """
    np.random.seed(seed)

    dt      = T / n_steps
    Z       = np.random.standard_normal((n_steps, n_paths))

    # GBM increment: exp((r - σ²/2)dt + σ√dt × Z)
    increments = np.exp(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    )

    # Build price paths
    paths       = np.zeros((n_steps + 1, n_paths))
    paths[0]    = S
    for t in range(1, n_steps + 1):
        paths[t] = paths[t - 1] * increments[t - 1]

    return paths


def mc_naive(S, K, T, r, sigma, n_paths=10000,
             n_steps=252, seed=42, option_type="call"):
    """
    Naive Monte Carlo pricing — baseline approach.

    Algorithm:
    1. Simulate n_paths terminal stock prices
    2. Compute payoff for each path
    3. Average payoffs and discount to today

    Price = e^(-rT) × E[max(S_T - K, 0)]

    Returns price and 95% confidence interval.
    """
    paths = simulate_gbm(S, r, sigma, T, n_paths, n_steps, seed)
    ST    = paths[-1]   # terminal prices only

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    discount   = np.exp(-r * T)
    price      = discount * np.mean(payoffs)
    std_error  = discount * np.std(payoffs) / np.sqrt(n_paths)

    # 95% confidence interval
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return {
        "price"    : price,
        "std_error": std_error,
        "ci_lower" : ci_lower,
        "ci_upper" : ci_upper,
        "ci_width" : ci_upper - ci_lower,
    }


def mc_antithetic(S, K, T, r, sigma, n_paths=10000,
                  n_steps=252, seed=42, option_type="call"):
    """
    Monte Carlo with antithetic variates variance reduction.

    Key insight: if Z generates a path, then -Z generates a
    mirror path. These two paths are negatively correlated —
    when one is high, the other is low.

    Averaging paired payoffs reduces variance:
        Var(mean(X, X')) = Var(X)/2 + Cov(X,X')/2

    Since Cov(X, X') < 0, variance is reduced significantly.
    Achieves ~50% variance reduction at no extra cost
    (uses same number of random numbers).

    Algorithm:
    1. Generate n_paths/2 random paths using Z
    2. Generate n_paths/2 mirror paths using -Z
    3. Average payoff of each paired (Z, -Z) simulation
    4. Average across all pairs and discount
    """
    np.random.seed(seed)
    half    = n_paths // 2
    dt      = T / n_steps

    # Generate random shocks
    Z       = np.random.standard_normal((n_steps, half))

    # Build paths and antithetic paths
    def build_paths(shocks):
        inc    = np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
        )
        paths       = np.zeros((n_steps + 1, half))
        paths[0]    = S
        for t in range(1, n_steps + 1):
            paths[t] = paths[t - 1] * inc[t - 1]
        return paths[-1]

    ST_orig = build_paths(Z)
    ST_anti = build_paths(-Z)

    if option_type == "call":
        pay_orig = np.maximum(ST_orig - K, 0)
        pay_anti = np.maximum(ST_anti - K, 0)
    else:
        pay_orig = np.maximum(K - ST_orig, 0)
        pay_anti = np.maximum(K - ST_anti, 0)

    # Average paired payoffs
    paired_payoffs = (pay_orig + pay_anti) / 2

    discount  = np.exp(-r * T)
    price     = discount * np.mean(paired_payoffs)
    std_error = discount * np.std(paired_payoffs) / np.sqrt(half)

    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return {
        "price"    : price,
        "std_error": std_error,
        "ci_lower" : ci_lower,
        "ci_upper" : ci_upper,
        "ci_width" : ci_upper - ci_lower,
    }


def mc_control_variate(S, K, T, r, sigma, n_paths=10000,
                       n_steps=252, seed=42, option_type="call"):
    """
    Monte Carlo with control variate variance reduction.

    Uses Black-Scholes price as a control variate — a quantity
    whose true value is known analytically.

    Key insight: the MC estimate of BS price and the MC estimate
    of our option price are correlated. We can exploit this
    correlation to reduce variance.

    Adjusted estimator:
        Price* = Price_MC - β × (BS_MC - BS_true)

    Where:
        β = Cov(payoff, control) / Var(control)
        BS_MC   = MC estimate of BS price (same paths)
        BS_true = exact BS price (known)

    If our MC is too high, BS_MC will also tend to be too high,
    and the correction pulls our estimate down. This can reduce
    variance by 90%+ for near-ATM options.
    """
    from black_scholes import call_price, put_price

    np.random.seed(seed)
    dt  = T / n_steps
    Z   = np.random.standard_normal((n_steps, n_paths))

    # Build paths
    inc         = np.exp(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    )
    paths       = np.zeros((n_steps + 1, n_paths))
    paths[0]    = S
    for t in range(1, n_steps + 1):
        paths[t] = paths[t - 1] * inc[t - 1]

    ST = paths[-1]

    # Target payoffs
    if option_type == "call":
        payoffs  = np.maximum(ST - K, 0)
        bs_true  = call_price(S, K, T, r, sigma)
        # Control: use ATM call as control (high correlation with target)
        controls = np.maximum(ST - K, 0)
    else:
        payoffs  = np.maximum(K - ST, 0)
        bs_true  = put_price(S, K, T, r, sigma)
        controls = np.maximum(K - ST, 0)

    discount    = np.exp(-r * T)
    bs_mc       = discount * np.mean(controls)

    # Compute optimal beta
    cov_matrix  = np.cov(payoffs, controls)
    beta        = cov_matrix[0, 1] / cov_matrix[1, 1] \
                  if cov_matrix[1, 1] > 0 else 1.0

    # Control variate adjusted payoffs
    adjusted    = payoffs - beta * (controls - bs_true / discount)
    price       = discount * np.mean(adjusted)
    std_error   = discount * np.std(adjusted) / np.sqrt(n_paths)

    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return {
        "price"    : price,
        "std_error": std_error,
        "ci_lower" : ci_lower,
        "ci_upper" : ci_upper,
        "ci_width" : ci_upper - ci_lower,
        "beta"     : round(beta, 4),
    }


def convergence_analysis(S, K, T, r, sigma,
                          path_counts=[100, 500, 1000,
                                       5000, 10000, 50000],
                          option_type="call", seed=42):
    """
    Show how MC price converges as number of paths increases.

    Demonstrates the 1/√N convergence rate — to halve the error,
    you need 4x more paths. Compares all three methods.

    Returns DataFrame with price estimates and errors per method.
    """
    from black_scholes import call_price, put_price

    bs_price = call_price(S, K, T, r, sigma) if option_type == "call" \
               else put_price(S, K, T, r, sigma)

    results = []
    for n in path_counts:
        naive = mc_naive(S, K, T, r, sigma, n, seed=seed,
                         option_type=option_type)
        anti  = mc_antithetic(S, K, T, r, sigma, n, seed=seed,
                               option_type=option_type)
        ctrl  = mc_control_variate(S, K, T, r, sigma, n, seed=seed,
                                    option_type=option_type)

        results.append({
            "N paths"       : n,
            "Naive"         : round(naive["price"], 4),
            "Naive SE"      : round(naive["std_error"], 4),
            "Antithetic"    : round(anti["price"], 4),
            "Antithetic SE" : round(anti["std_error"], 4),
            "Control"       : round(ctrl["price"], 4),
            "Control SE"    : round(ctrl["std_error"], 4),
            "BS True"       : round(bs_price, 4),
        })

    return pd.DataFrame(results)


def asian_option_price(S, K, T, r, sigma, n_paths=10000,
                       n_steps=252, seed=42, option_type="call"):
    """
    Price an Asian option using Monte Carlo.

    Asian options pay based on the AVERAGE stock price over the
    option's life, not the terminal price. This cannot be priced
    analytically — Monte Carlo is the natural approach.

    Payoff = max(S_avg - K, 0) for Asian call
           = max(K - S_avg, 0) for Asian put

    Asian options are cheaper than European because averaging
    reduces the effective volatility. Commonly used in commodities
    and structured products to reduce manipulation risk.
    """
    paths = simulate_gbm(S, r, sigma, T, n_paths, n_steps, seed)

    # Average price along each path (arithmetic mean)
    S_avg = paths.mean(axis=0)

    if option_type == "call":
        payoffs = np.maximum(S_avg - K, 0)
    else:
        payoffs = np.maximum(K - S_avg, 0)

    discount  = np.exp(-r * T)
    price     = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)

    return {
        "Asian price"   : round(price, 4),
        "European price": round(
            mc_naive(S, K, T, r, sigma, n_paths,
                     n_steps, seed, option_type)["price"], 4
        ),
        "Asian discount": round(
            1 - price / mc_naive(S, K, T, r, sigma, n_paths,
                                  n_steps, seed, option_type)["price"],
            4
        ),
        "Std Error"     : round(std_error, 6),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import get_model_inputs
    from black_scholes import call_price

    inputs = get_model_inputs(ticker="VCB", vol_window=30)
    S      = inputs["S"]
    r      = inputs["r"]
    sigma  = inputs["sigma"]
    K      = S        # ATM
    T      = 0.25     # 3 months

    bs_price = call_price(S, K, T, r, sigma)

    print(f"\n{'='*60}")
    print(f"  Monte Carlo: ATM Call on {inputs['ticker']}")
    print(f"{'='*60}")
    print(f"  S={S:.1f}  K={K:.1f}  T={T}y  "
          f"r={r*100:.1f}%  σ={sigma*100:.2f}%")
    print(f"  BS True Price: {bs_price:.4f}")

    # Compare three methods
    print(f"\nPricing Methods Comparison (10,000 paths):")
    print(f"  {'Method':<20} {'Price':>8} {'Std Error':>10} "
          f"{'95% CI':>20} {'CI Width':>10}")
    print(f"  {'─'*70}")

    for name, fn in [
        ("Naive MC",    mc_naive),
        ("Antithetic",  mc_antithetic),
        ("Control",     mc_control_variate),
    ]:
        res = fn(S, K, T, r, sigma, n_paths=10000, option_type="call")
        print(f"  {name:<20} {res['price']:>8.4f} "
              f"{res['std_error']:>10.4f} "
              f"[{res['ci_lower']:>8.4f}, {res['ci_upper']:>8.4f}] "
              f"{res['ci_width']:>10.4f}")

    print(f"  {'BS Analytical':<20} {bs_price:>8.4f} "
          f"{'—':>10} {'—':>20} {'0':>10}")

    # Convergence
    print(f"\nConvergence Analysis (ATM Call):")
    conv = convergence_analysis(S, K, T, r, sigma,
                                 path_counts=[100, 1000, 5000, 10000, 50000])
    print(conv.to_string(index=False))

    # Asian option
    print(f"\nAsian vs European Call:")
    asian = asian_option_price(S, K, T, r, sigma, n_paths=10000)
    for k, v in asian.items():
        print(f"  {k:<20} {v}")