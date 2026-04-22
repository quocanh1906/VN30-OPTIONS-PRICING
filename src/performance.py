# src/performance.py

"""
performance.py — Visualisation and reporting for options pricing engine.

Produces publication-quality charts for all five modules:
1. Black-Scholes price and Greeks surfaces
2. Binomial tree convergence
3. Monte Carlo convergence and path visualisation
4. Implied volatility smile and surface
5. Delta hedging P&L distribution and frequency comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")


def plot_greeks(S, r, sigma, option_type="call"):
    """
    Plot all five Greeks as a function of spot price.
    Shows how Greeks evolve from deep OTM to deep ITM.
    """
    from black_scholes import (call_price, put_price,
                                delta, gamma, vega, theta, rho)

    K    = S          # ATM strike
    T    = 0.25       # 3 months
    spots = np.linspace(S * 0.6, S * 1.4, 200)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Black-Scholes Greeks — {option_type.capitalize()} Option "
                 f"(K={K:.1f}, T=3M, σ={sigma*100:.1f}%)",
                 fontsize=13, fontweight="bold")

    # Option price
    ax = axes[0][0]
    pricer = call_price if option_type == "call" else put_price
    prices = [pricer(s, K, T, r, sigma) for s in spots]
    ax.plot(spots, prices, color="steelblue", linewidth=2)
    ax.axvline(x=K, color="red", linestyle="--", alpha=0.5, label="ATM")
    ax.set_title("Option Price")
    ax.set_xlabel("Spot Price")
    ax.legend()
    ax.grid(alpha=0.3)

    # Delta
    ax = axes[0][1]
    deltas = [delta(s, K, T, r, sigma, option_type) for s in spots]
    ax.plot(spots, deltas, color="darkorange", linewidth=2)
    ax.axvline(x=K, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.5)
    ax.set_title("Delta (Δ)")
    ax.set_xlabel("Spot Price")
    ax.grid(alpha=0.3)

    # Gamma
    ax = axes[0][2]
    gammas = [gamma(s, K, T, r, sigma) for s in spots]
    ax.plot(spots, gammas, color="green", linewidth=2)
    ax.axvline(x=K, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Gamma (Γ) — peaks ATM")
    ax.set_xlabel("Spot Price")
    ax.grid(alpha=0.3)

    # Vega
    ax = axes[1][0]
    vegas = [vega(s, K, T, r, sigma) for s in spots]
    ax.plot(spots, vegas, color="purple", linewidth=2)
    ax.axvline(x=K, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Vega (ν) — sensitivity to vol")
    ax.set_xlabel("Spot Price")
    ax.grid(alpha=0.3)

    # Theta
    ax = axes[1][1]
    thetas = [theta(s, K, T, r, sigma, option_type) for s in spots]
    ax.plot(spots, thetas, color="brown", linewidth=2)
    ax.axvline(x=K, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title("Theta (Θ) — time decay per day")
    ax.set_xlabel("Spot Price")
    ax.grid(alpha=0.3)

    # Rho
    ax = axes[1][2]
    from black_scholes import rho
    rhos = [rho(s, K, T, r, sigma, option_type) for s in spots]
    ax.plot(spots, rhos, color="teal", linewidth=2)
    ax.axvline(x=K, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Rho (ρ) — sensitivity to rate")
    ax.set_xlabel("Spot Price")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/greeks.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to output/greeks.png")


def plot_binomial_convergence(S, K, T, r, sigma):
    """
    Plot binomial tree convergence to Black-Scholes.
    Shows oscillating convergence pattern characteristic of CRR.
    """
    from binomial import convergence_analysis

    steps = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    conv  = convergence_analysis(S, K, T, r, sigma, steps)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Binomial Tree Convergence to Black-Scholes",
                 fontsize=13, fontweight="bold")

    # Price convergence
    ax = axes[0]
    ax.plot(conv["N steps"], conv["Binomial Price"],
            color="steelblue", linewidth=1.5,
            marker="o", markersize=4, label="Binomial")
    ax.axhline(y=conv["BS Price"].iloc[0], color="red",
               linestyle="--", linewidth=1.5, label="BS Analytical")
    ax.set_xlabel("Number of Steps (N)")
    ax.set_ylabel("Option Price")
    ax.set_title("Price Convergence")
    ax.legend()
    ax.grid(alpha=0.3)

    # Error convergence
    ax = axes[1]
    ax.plot(conv["N steps"], conv["Error (%)"].abs(),
            color="darkorange", linewidth=1.5,
            marker="o", markersize=4)
    ax.set_xlabel("Number of Steps (N)")
    ax.set_ylabel("Absolute Error (%)")
    ax.set_title("Error vs N Steps (log scale)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/binomial_convergence.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to output/binomial_convergence.png")


def plot_mc_paths(S, r, sigma, T=0.25, n_paths=50, seed=42):
    """
    Visualise Monte Carlo price paths.
    Shows the fan of possible stock price trajectories.
    """
    from monte_carlo import simulate_gbm

    paths = simulate_gbm(S, r, sigma, T,
                          n_paths=n_paths,
                          n_steps=int(T*252),
                          seed=seed)

    t = np.linspace(0, T, paths.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Monte Carlo: {n_paths} GBM Paths (σ={sigma*100:.1f}%)",
                 fontsize=13, fontweight="bold")

    # Path trajectories
    ax = axes[0]
    for i in range(n_paths):
        ax.plot(t, paths[:, i], alpha=0.3, linewidth=0.8,
                color="steelblue")
    ax.axhline(y=S, color="black", linestyle="--",
               linewidth=1, label="Initial price")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price")
    ax.set_title("Price Paths")
    ax.legend()
    ax.grid(alpha=0.3)

    # Terminal distribution
    ax = axes[1]
    ST = paths[-1, :]
    ax.hist(ST, bins=30, density=True, color="steelblue",
            alpha=0.7, label="Simulated")

    # Overlay log-normal PDF
    from scipy.stats import lognorm
    mu_ln  = np.log(S) + (r - 0.5*sigma**2) * T
    sig_ln = sigma * np.sqrt(T)
    x      = np.linspace(ST.min(), ST.max(), 200)
    ax.plot(x, lognorm.pdf(x, s=sig_ln, scale=np.exp(mu_ln)),
            color="red", linewidth=2, label="Log-normal PDF")
    ax.axvline(x=S, color="black", linestyle="--",
               linewidth=1, label="Initial price")
    ax.set_xlabel("Terminal Price")
    ax.set_ylabel("Density")
    ax.set_title("Terminal Price Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/mc_paths.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to output/mc_paths.png")


def plot_vol_surface(S, r, sigma):
    """
    Plot 3D implied volatility surface.
    Shows smile across strikes and term structure across maturities.
    """
    from implied_vol import build_vol_surface

    moneyness  = [0.80, 0.85, 0.90, 0.95, 1.0,
                  1.05, 1.10, 1.15, 1.20]
    maturities = [1/12, 3/12, 6/12, 1.0, 2.0]

    iv_surf, _ = build_vol_surface(
        S, r,
        atm_vol     = sigma,
        skew        = -0.10,
        smile_curve = 0.05,
        moneyness   = moneyness,
        maturities  = maturities,
    )

    # Clean NaN
    iv_surf = iv_surf.bfill().ffill()

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Implied Volatility Surface — VCB Options (Synthetic)",
                 fontsize=13, fontweight="bold")

    # 3D surface
    ax1 = fig.add_subplot(121, projection="3d")
    X   = np.array([int(t*12) for t in maturities])
    Y   = np.array(moneyness) * 100
    Z   = iv_surf.values.astype(float) * 100

    XX, YY = np.meshgrid(X, Y)
    surf = ax1.plot_surface(XX, YY, Z, cmap="RdYlGn_r", alpha=0.8)
    ax1.set_xlabel("Maturity (months)")
    ax1.set_ylabel("Moneyness (%)")
    ax1.set_zlabel("Implied Vol (%)")
    ax1.set_title("3D Vol Surface")
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # 2D smile for each maturity
    ax2 = fig.add_subplot(122)
    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    for i, (col, color) in enumerate(zip(iv_surf.columns, colors)):
        vals = iv_surf[col].values.astype(float) * 100
        ax2.plot(Y, vals, color=color, linewidth=1.8,
                 marker="o", markersize=4, label=col)

    ax2.axvline(x=100, color="black", linestyle="--",
                alpha=0.5, label="ATM")
    ax2.set_xlabel("Moneyness (% of spot)")
    ax2.set_ylabel("Implied Vol (%)")
    ax2.set_title("Volatility Smile by Maturity")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/vol_surface.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to output/vol_surface.png")


def plot_delta_hedge(S, K, T, r, sigma):
    """
    Plot delta hedging results:
    1. P&L distribution across 500 paths
    2. Rebalancing frequency vs cost tradeoff
    3. Vol mismatch P&L sensitivity
    """
    from delta_hedge import (simulate_hedge,
                              rebalance_frequency_comparison,
                              vol_mismatch_analysis)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Delta Hedging Analysis — Short ATM Call on VCB",
                 fontsize=13, fontweight="bold")

    # P&L distribution
    ax = axes[0]
    _, summary = simulate_hedge(S, K, T, r, sigma,
                                 rebalance_freq="daily",
                                 n_paths=500, seed=42,
                                 include_costs=True)

    # Regenerate P&Ls for histogram
    pnls = []
    np.random.seed(42)
    n_steps = int(T * 252)
    dt      = T / n_steps
    Z       = np.random.standard_normal((n_steps, 500))
    paths   = np.zeros((n_steps + 1, 500))
    paths[0] = S
    for t in range(1, n_steps + 1):
        paths[t] = paths[t-1] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t-1]
        )

    from black_scholes import call_price as cp, delta as dl
    from delta_hedge import COMMISSION, SALES_TAX

    for path_idx in range(500):
        S_path = paths[:, path_idx]
        cash   = cp(S, K, T, r, sigma)
        shares = dl(S, K, T, r, sigma, "call")
        cash  -= shares * S + shares * S * COMMISSION

        for step in range(1, n_steps + 1):
            S_t   = S_path[step]
            T_rem = T - step * dt
            cash *= np.exp(r * dt)
            new_delta = dl(S_t, K, max(T_rem,1e-8),
                           r, sigma, "call") if T_rem > 1e-8 \
                        else (1.0 if S_t > K else 0.0)
            dc    = new_delta - shares
            cost  = abs(dc) * S_t * (COMMISSION +
                    (SALES_TAX if dc < 0 else 0))
            cash -= dc * S_t + cost
            shares = new_delta

        cash += shares * S_path[-1] * (1 - COMMISSION - SALES_TAX)
        cash -= max(S_path[-1] - K, 0)
        pnls.append(cash)

    pnls = np.array(pnls)
    ax.hist(pnls, bins=40, color="steelblue", alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(x=np.mean(pnls), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean: {np.mean(pnls):.3f}")
    ax.set_xlabel("Hedging P&L")
    ax.set_ylabel("Frequency")
    ax.set_title("P&L Distribution (500 paths, daily)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Rebalancing frequency comparison
    ax = axes[1]
    freq_df = rebalance_frequency_comparison(S, K, T, r, sigma, 500)
    x       = np.arange(len(freq_df))
    width   = 0.35
    ax.bar(x - width/2, freq_df["Std P&L (no cost)"],
           width, label="Std (no cost)", color="steelblue", alpha=0.7)
    ax.bar(x + width/2, freq_df["Std P&L (w/ cost)"],
           width, label="Std (w/ cost)", color="darkorange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(freq_df["Frequency"])
    ax.set_ylabel("P&L Std Dev")
    ax.set_title("Hedging Error by Rebalancing Frequency")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Vol mismatch
    ax = axes[2]
    vm_df = vol_mismatch_analysis(
        S, K, T, r,
        implied_vol   = sigma,
        realised_vols = [sigma * x for x in [0.5, 0.75, 1.0, 1.25, 1.5]],
        n_paths       = 500,
    )
    colors = ["green" if v > 0 else "red"
              for v in vm_df["Mean P&L"]]
    ax.bar(vm_df["Vol Spread (%)"], vm_df["Mean P&L"],
           color=colors, alpha=0.8, width=1.2)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Vol Spread: Implied - Realised (%)")
    ax.set_ylabel("Mean P&L")
    ax.set_title("P&L vs Volatility Mismatch")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("output/delta_hedge.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to output/delta_hedge.png")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import get_model_inputs

    inputs = get_model_inputs(ticker="VCB", vol_window=30)
    S      = inputs["S"]
    r      = inputs["r"]
    sigma  = inputs["sigma"]
    K      = S
    T      = 0.25

    os.makedirs("output", exist_ok=True)

    print("\nGenerating all charts...")
    plot_greeks(S, r, sigma, "call")
    plot_binomial_convergence(S, K, T, r, sigma)
    plot_mc_paths(S, r, sigma, T)
    plot_vol_surface(S, r, sigma)
    plot_delta_hedge(S, K, T, r, sigma)
    print("\nAll charts saved to output/")