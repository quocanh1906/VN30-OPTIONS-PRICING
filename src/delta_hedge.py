"""
delta_hedge.py — Delta hedging simulation and P&L analysis.

Demonstrates the practical connection between theoretical options
pricing and real risk management.

Scenario: an options dealer sells a call option and delta hedges
the position by buying/selling the underlying stock dynamically.

In continuous time (Black-Scholes world):
    P&L = 0 exactly (perfect hedge)

In discrete time (reality):
    P&L ≠ 0 — hedging error accumulates due to:
    1. Gamma risk — delta changes between rebalances
    2. Transaction costs — every rebalance incurs costs
    3. Volatility mismatch — realised vol ≠ implied vol

The P&L of a delta-hedged short call is driven by:
    Daily P&L ≈ ½ × Γ × (ΔS)² - Θ × Δt

    Gamma term: you LOSE when stock moves a lot (short gamma)
    Theta term: you GAIN from time decay (short option = long theta)

This is the fundamental tradeoff:
    Short option → collect premium + earn theta → but exposed to gamma

Key insight for risk management:
    If realised vol > implied vol: net loss (paid too little premium)
    If realised vol < implied vol: net profit (charged too much premium)
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from black_scholes import call_price, delta, gamma, theta


# Vietnamese market transaction costs
COMMISSION = 0.00125   # 0.125% one-way
SALES_TAX  = 0.001     # 0.10% on sells only


def simulate_hedge(S0, K, T, r, sigma,
                   rebalance_freq="daily",
                   n_paths=1,
                   seed=42,
                   include_costs=True):
    """
    Simulate delta hedging of a short call position.

    The dealer:
    1. Sells 1 call option — receives premium
    2. Buys delta shares of stock — pays cost
    3. Rebalances delta every period
    4. At expiry: settles the option and unwinds hedge

    P&L = Premium received - Hedging costs - Option payoff at expiry

    Parameters
    ----------
    S0             : initial spot price
    K              : strike price
    T              : time to expiry in years
    r              : risk-free rate
    sigma          : implied volatility (used for pricing and delta)
    rebalance_freq : 'daily', 'weekly', 'monthly'
    n_paths        : number of stock price paths to simulate
    seed           : random seed
    include_costs  : whether to include transaction costs

    Returns
    -------
    DataFrame with daily hedge record for each path
    dict with summary statistics across paths
    """
    np.random.seed(seed)

    # Time grid
    freq_map = {"daily": 252, "weekly": 52, "monthly": 12}
    steps_per_year = freq_map.get(rebalance_freq, 252)
    n_steps = max(int(T * steps_per_year), 1)
    dt      = T / n_steps

    # Simulate stock price paths (GBM)
    Z     = np.random.standard_normal((n_steps, n_paths))
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    for t in range(1, n_steps + 1):
        paths[t] = paths[t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1]
        )

    all_records = []
    final_pnls  = []

    for path_idx in range(n_paths):
        S_path  = paths[:, path_idx]
        records = []

        # t=0: sell call, buy delta shares
        t0      = 0
        S_t     = S_path[0]
        T_rem   = T
        C_t     = call_price(S_t, K, T_rem, r, sigma)
        delta_t = delta(S_t, K, T_rem, r, sigma, "call")
        gamma_t = gamma(S_t, K, T_rem, r, sigma)
        theta_t = theta(S_t, K, T_rem, r, sigma, "call")

        # Cash flows at initiation
        premium_received = C_t          # received for selling call
        stock_bought     = delta_t * S_t
        cost_buy         = stock_bought * COMMISSION if include_costs else 0
        cash             = premium_received - stock_bought - cost_buy

        records.append({
            "step"          : 0,
            "S"             : round(S_t, 4),
            "T_remaining"   : round(T_rem, 4),
            "delta"         : round(delta_t, 4),
            "gamma"         : round(gamma_t, 6),
            "theta"         : round(theta_t, 6),
            "option_value"  : round(C_t, 4),
            "shares_held"   : round(delta_t, 4),
            "cash"          : round(cash, 4),
            "hedge_cost"    : round(cost_buy, 4),
            "portfolio_value": round(delta_t * S_t + cash, 4),
        })

        shares_held = delta_t

        # Rebalancing loop
        for step in range(1, n_steps + 1):
            S_t   = S_path[step]
            T_rem = T - step * dt

            if T_rem > 1e-8:
                C_t     = call_price(S_t, K, T_rem, r, sigma)
                delta_t = delta(S_t, K, T_rem, r, sigma, "call")
                gamma_t = gamma(S_t, K, T_rem, r, sigma)
                theta_t = theta(S_t, K, T_rem, r, sigma, "call")
            else:
                # At expiry
                C_t     = max(S_t - K, 0)
                delta_t = 1.0 if S_t > K else 0.0
                gamma_t = 0.0
                theta_t = 0.0

            # Accrue interest on cash
            cash *= np.exp(r * dt)

            # Rebalance: buy/sell to match new delta
            delta_change = delta_t - shares_held
            trade_value  = abs(delta_change) * S_t

            if include_costs:
                if delta_change > 0:  # buying shares
                    cost = trade_value * COMMISSION
                else:                 # selling shares
                    cost = trade_value * (COMMISSION + SALES_TAX)
            else:
                cost = 0

            cash         -= delta_change * S_t + cost
            shares_held   = delta_t

            records.append({
                "step"          : step,
                "S"             : round(S_t, 4),
                "T_remaining"   : round(max(T_rem, 0), 4),
                "delta"         : round(delta_t, 4),
                "gamma"         : round(gamma_t, 6),
                "theta"         : round(theta_t, 6),
                "option_value"  : round(C_t, 4),
                "shares_held"   : round(shares_held, 4),
                "cash"          : round(cash, 4),
                "hedge_cost"    : round(cost, 4),
                "portfolio_value": round(shares_held * S_t + cash, 4),
            })

        # Final P&L
        S_final      = S_path[-1]
        option_payoff = max(S_final - K, 0)

        # Unwind hedge
        unwind_value = shares_held * S_final
        unwind_cost  = unwind_value * (COMMISSION + SALES_TAX) \
                       if include_costs else 0
        final_cash   = cash + unwind_value - unwind_cost - option_payoff

        # Total P&L = final cash position
        # (started with 0 net cash — premium offset stock purchase)
        final_pnl = final_cash
        final_pnls.append(final_pnl)

        df = pd.DataFrame(records)
        df["path"] = path_idx
        all_records.append(df)

    all_df = pd.concat(all_records, ignore_index=True)

    pnls = np.array(final_pnls)
    summary = {
        "n_paths"         : n_paths,
        "rebalance_freq"  : rebalance_freq,
        "include_costs"   : include_costs,
        "mean_pnl"        : round(np.mean(pnls), 4),
        "std_pnl"         : round(np.std(pnls), 4),
        "min_pnl"         : round(np.min(pnls), 4),
        "max_pnl"         : round(np.max(pnls), 4),
        "pct_profitable"  : round((pnls > 0).mean() * 100, 1),
        "initial_premium" : round(call_price(S0, K, T, r, sigma), 4),
    }

    return all_df, summary


def rebalance_frequency_comparison(S, K, T, r, sigma,
                                    n_paths=500, seed=42):
    """
    Compare hedging error across rebalancing frequencies.

    More frequent rebalancing:
    - Reduces gamma risk (smaller delta moves between rebalances)
    - Increases transaction costs
    - Net effect depends on gamma × (ΔS)² vs cost per trade

    For low-volatility stocks, costs dominate → less frequent is better.
    For high-volatility stocks, gamma dominates → more frequent is better.

    Returns DataFrame comparing mean P&L and std across frequencies.
    """
    results = []
    freqs   = ["daily", "weekly", "monthly"]

    for freq in freqs:
        # Without costs
        _, summary_no_cost = simulate_hedge(
            S, K, T, r, sigma,
            rebalance_freq = freq,
            n_paths        = n_paths,
            seed           = seed,
            include_costs  = False,
        )

        # With costs
        _, summary_cost = simulate_hedge(
            S, K, T, r, sigma,
            rebalance_freq = freq,
            n_paths        = n_paths,
            seed           = seed,
            include_costs  = True,
        )

        results.append({
            "Frequency"          : freq,
            "Mean P&L (no cost)" : summary_no_cost["mean_pnl"],
            "Std P&L (no cost)"  : summary_no_cost["std_pnl"],
            "Mean P&L (w/ cost)" : summary_cost["mean_pnl"],
            "Std P&L (w/ cost)"  : summary_cost["std_pnl"],
            "Cost drag"          : round(
                summary_no_cost["mean_pnl"] -
                summary_cost["mean_pnl"], 4
            ),
        })

    return pd.DataFrame(results)


def vol_mismatch_analysis(S, K, T, r,
                           implied_vol=0.20,
                           realised_vols=[0.10, 0.15, 0.20, 0.25, 0.30],
                           n_paths=500, seed=42):
    """
    Show P&L sensitivity to realised vs implied vol mismatch.

    Key insight: a delta-hedged short call profits when:
        realised vol < implied vol (overpriced option)

    and loses when:
        realised vol > implied vol (underpriced option)

    This is the core of volatility trading — options dealers are
    essentially trading the spread between implied and realised vol.

    Parameters
    ----------
    implied_vol   : vol used for pricing and delta hedging
    realised_vols : actual vol of stock price paths simulated
    """
    results = []

    for rv in realised_vols:
        # Simulate paths with realised vol
        # but hedge using implied vol
        np.random.seed(seed)
        n_steps = int(T * 252)
        dt      = T / n_steps
        Z       = np.random.standard_normal((n_steps, n_paths))

        paths   = np.zeros((n_steps + 1, n_paths))
        paths[0] = S
        for t in range(1, n_steps + 1):
            paths[t] = paths[t-1] * np.exp(
                (r - 0.5 * rv**2) * dt + rv * np.sqrt(dt) * Z[t-1]
            )

        pnls = []
        for path_idx in range(n_paths):
            S_path = paths[:, path_idx]
            cash   = call_price(S, K, T, r, implied_vol)  # premium
            shares = delta(S, K, T, r, implied_vol, "call")
            cash  -= shares * S

            for step in range(1, n_steps + 1):
                S_t   = S_path[step]
                T_rem = T - step * dt
                cash *= np.exp(r * dt)

                if T_rem > 1e-8:
                    new_delta = delta(S_t, K, T_rem, r, implied_vol, "call")
                else:
                    new_delta = 1.0 if S_t > K else 0.0

                cash  -= (new_delta - shares) * S_t
                shares = new_delta

            S_final = S_path[-1]
            cash   += shares * S_final - max(S_final - K, 0)
            pnls.append(cash)

        pnls = np.array(pnls)
        results.append({
            "Realised Vol (%)" : round(rv * 100, 1),
            "Implied Vol (%)"  : round(implied_vol * 100, 1),
            "Vol Spread (%)"   : round((implied_vol - rv) * 100, 1),
            "Mean P&L"         : round(np.mean(pnls), 4),
            "Std P&L"          : round(np.std(pnls), 4),
            "% Profitable"     : round((pnls > 0).mean() * 100, 1),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import get_model_inputs

    inputs = get_model_inputs(ticker="VCB", vol_window=30)
    S      = inputs["S"]
    r      = inputs["r"]
    sigma  = inputs["sigma"]
    K      = S        # ATM
    T      = 0.25     # 3 months

    print(f"\n{'='*60}")
    print(f"  Delta Hedging Simulation — {inputs['ticker']}")
    print(f"{'='*60}")
    print(f"  Short ATM call: S={S:.1f} K={K:.1f} "
          f"T={T}y σ={sigma*100:.2f}%")
    print(f"  Premium received: {call_price(S, K, T, r, sigma):.4f}")

    # Single path daily hedge record
    print(f"\nSingle path daily hedge (first 10 steps):")
    df, summary = simulate_hedge(
        S, K, T, r, sigma,
        rebalance_freq="daily",
        n_paths=1,
        seed=42,
        include_costs=True,
    )
    print(df[["step", "S", "delta", "gamma",
              "shares_held", "cash",
              "portfolio_value"]].head(10).to_string(index=False))

    print(f"\nSingle path summary:")
    for k, v in summary.items():
        print(f"  {k:<25} {v}")

    # Rebalancing frequency comparison
    print(f"\nRebalancing Frequency Comparison (500 paths):")
    freq_df = rebalance_frequency_comparison(S, K, T, r, sigma,
                                              n_paths=500)
    print(freq_df.to_string(index=False))

    # Vol mismatch analysis
    print(f"\nVol Mismatch Analysis (implied={sigma*100:.1f}%):")
    vm_df = vol_mismatch_analysis(
        S, K, T, r,
        implied_vol   = sigma,
        realised_vols = [sigma * x for x in
                         [0.5, 0.75, 1.0, 1.25, 1.5]],
        n_paths       = 500,
    )
    print(vm_df.to_string(index=False))

    # Save
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/hedge_record.csv", index=False)
    freq_df.to_csv("output/frequency_comparison.csv", index=False)
    vm_df.to_csv("output/vol_mismatch.csv", index=False)
    print("\nSaved to output/")