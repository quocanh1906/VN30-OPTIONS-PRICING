"""
main.py — Full options pricing pipeline.

Runs all five modules in sequence and produces a complete
pricing and risk analysis report for VCB options.

Output:
    output/greeks.png
    output/binomial_convergence.png
    output/mc_paths.png
    output/vol_surface.png
    output/delta_hedge.png
    output/pricing_summary.csv
"""

import sys
import os
sys.path.insert(0, "src")

import pandas as pd
import numpy as np

from data import get_model_inputs, get_option_grid
from black_scholes import (call_price, put_price,
                            all_greeks, price_surface,
                            put_call_parity_check)
from binomial import (european_price, american_price,
                      early_exercise_premium, convergence_analysis)
from monte_carlo import mc_naive, mc_antithetic, mc_control_variate
from implied_vol import compute_smile, build_vol_surface
from delta_hedge import (simulate_hedge, rebalance_frequency_comparison,
                          vol_mismatch_analysis)
from performance import (plot_greeks, plot_binomial_convergence,
                          plot_mc_paths, plot_vol_surface,
                          plot_delta_hedge)

print("=" * 60)
print("VN30 Options Pricing Engine")
print("=" * 60)

# ── 1. Load market data ─────────────────────────────────────────────────────────
print("\nStep 1: Loading market data...")
inputs = get_model_inputs(ticker="VCB", vol_window=30)
S      = inputs["S"]
r      = inputs["r"]
sigma  = inputs["sigma"]
K      = S        # ATM strike
T      = 0.25     # 3-month option

print(f"\nPricing: ATM {T*12:.0f}-month option on VCB")
print(f"  S={S:.1f}  K={K:.1f}  r={r*100:.1f}%  σ={sigma*100:.2f}%")

# ── 2. Black-Scholes ────────────────────────────────────────────────────────────
print("\nStep 2: Black-Scholes pricing...")
C_bs = call_price(S, K, T, r, sigma)
P_bs = put_price(S, K, T, r, sigma)

print(f"  Call = {C_bs:.4f}  Put = {P_bs:.4f}")

pcp = put_call_parity_check(S, K, T, r, sigma)
print(f"  Put-call parity error: {pcp['Parity error']:.2e}")

call_greeks = all_greeks(S, K, T, r, sigma, "call")
print(f"  Call Greeks: Δ={call_greeks['Delta']:.3f}  "
      f"Γ={call_greeks['Gamma']:.5f}  "
      f"ν={call_greeks['Vega']:.3f}  "
      f"Θ={call_greeks['Theta']:.4f}")

# ── 3. Binomial tree ────────────────────────────────────────────────────────────
print("\nStep 3: Binomial tree pricing...")
bin_call = european_price(S, K, T, r, sigma, N=500, option_type="call")
bin_put  = european_price(S, K, T, r, sigma, N=500, option_type="put")
am_put   = american_price(S, K, T, r, sigma, N=500, option_type="put")
eep      = early_exercise_premium(S, K, T, r, sigma)

print(f"  European Call = {bin_call:.4f}  (BS diff: {bin_call-C_bs:+.6f})")
print(f"  European Put  = {bin_put:.4f}  (BS diff: {bin_put-P_bs:+.6f})")
print(f"  American Put  = {am_put:.4f}")
print(f"  Early exercise premium = {eep['Early Exercise Premium']:.4f} "
      f"({eep['Premium as % of Euro']:.2f}% of European)")

# ── 4. Monte Carlo ──────────────────────────────────────────────────────────────
print("\nStep 4: Monte Carlo pricing (10,000 paths)...")
mc_naive_res = mc_naive(S, K, T, r, sigma, n_paths=10000, option_type="call")
mc_anti_res  = mc_antithetic(S, K, T, r, sigma,
                               n_paths=10000, option_type="call")
mc_ctrl_res  = mc_control_variate(S, K, T, r, sigma,
                                   n_paths=10000, option_type="call")

print(f"  Naive MC   : {mc_naive_res['price']:.4f} "
      f"± {mc_naive_res['std_error']:.4f}")
print(f"  Antithetic : {mc_anti_res['price']:.4f} "
      f"± {mc_anti_res['std_error']:.4f}")
print(f"  Control CV : {mc_ctrl_res['price']:.4f} "
      f"± {mc_ctrl_res['std_error']:.4f}")
print(f"  BS True    : {C_bs:.4f}")

# ── 5. Implied volatility ───────────────────────────────────────────────────────
print("\nStep 5: Implied volatility surface...")
iv_surf, _ = build_vol_surface(S, r,
                                atm_vol=sigma,
                                skew=-0.10,
                                smile_curve=0.05)
print("  Vol surface computed (synthetic smile)")
print(f"  ATM IV: {sigma*100:.2f}%")
print(f"  80% ATM IV (3M): {iv_surf['T=3m'].iloc[0]*100:.2f}% "
      f"(skew premium)")

# ── 6. Delta hedging ────────────────────────────────────────────────────────────
print("\nStep 6: Delta hedging simulation (500 paths)...")
freq_df = rebalance_frequency_comparison(S, K, T, r, sigma, n_paths=500)
print(freq_df[["Frequency", "Mean P&L (no cost)",
               "Mean P&L (w/ cost)", "Cost drag"]].to_string(index=False))

vm_df = vol_mismatch_analysis(S, K, T, r,
                               implied_vol=sigma,
                               realised_vols=[sigma*x
                               for x in [0.5, 0.75, 1.0, 1.25, 1.5]],
                               n_paths=500)
print(f"\n  Vol mismatch: when realised < implied → profitable")
print(vm_df[["Realised Vol (%)", "Vol Spread (%)",
             "Mean P&L", "% Profitable"]].to_string(index=False))

# ── 7. Save summary ─────────────────────────────────────────────────────────────
print("\nStep 7: Saving summary...")
os.makedirs("output", exist_ok=True)

summary = pd.DataFrame([{
    "Ticker"          : "VCB",
    "Spot"            : S,
    "Strike"          : K,
    "Maturity"        : T,
    "Risk-free rate"  : r,
    "Hist vol (30d)"  : sigma,
    "BS Call"         : C_bs,
    "BS Put"          : P_bs,
    "Binomial Call"   : bin_call,
    "American Put"    : am_put,
    "EEP"             : eep["Early Exercise Premium"],
    "MC Naive Call"   : mc_naive_res["price"],
    "MC Antithetic"   : mc_anti_res["price"],
    "MC Control"      : mc_ctrl_res["price"],
    "Call Delta"      : call_greeks["Delta"],
    "Call Gamma"      : call_greeks["Gamma"],
    "Call Vega"       : call_greeks["Vega"],
    "Call Theta"      : call_greeks["Theta"],
}])
summary.to_csv("output/pricing_summary.csv", index=False)
print("  Saved to output/pricing_summary.csv")

# ── 8. Generate all charts ──────────────────────────────────────────────────────
print("\nStep 8: Generating charts...")
plot_greeks(S, r, sigma, "call")
plot_binomial_convergence(S, K, T, r, sigma)
plot_mc_paths(S, r, sigma, T, n_paths=100)
plot_vol_surface(S, r, sigma)
plot_delta_hedge(S, K, T, r, sigma)

print("\nDone! All outputs saved to output/")