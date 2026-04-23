"""
main.py — Full options pricing pipeline.

Runs the three-pricer stack (Black–Scholes / binomial / Monte Carlo)
under two different volatility inputs — rolling historical σ and
GARCH(1,1) — and produces comparison outputs that highlight what
the choice of vol estimator actually buys you.

Pricers in use
--------------
  BS      : closed-form, constant σ          → needs a scalar
  Binomial: tree, constant σ                  → needs a scalar
  MC      : full GARCH path (Duan 1995)       → needs (ω, α, β, σ²_t)

The BS and binomial price lines can only reflect the vol through
their scalar input. They therefore use two "levels" of GARCH:
  Level 1 : GARCH Level-1 forecast σ          (average conditional
            variance over [0, T] annualised)
  Level 2 : only the MC pricer uses this — full day-by-day GARCH
            dynamics under the risk-neutral measure.

See README.md for the full theoretical walk-through.

Outputs
-------
  output/greeks.png
  output/binomial_convergence.png
  output/mc_paths.png
  output/vol_surface.png
  output/vol_comparison.png      ← historical vs GARCH
  output/endogenous_smile.png    ← GARCH smile from MC
  output/delta_hedge.png
  output/pricing_summary.csv
"""

import sys
import os
sys.path.insert(0, "src")

import pandas as pd

from data import get_model_inputs
from black_scholes import (call_price, put_price,
                            all_greeks, put_call_parity_check)
from binomial import (european_price, american_price,
                      early_exercise_premium)
from monte_carlo import (mc_naive, mc_antithetic, mc_control_variate,
                         mc_garch)
from implied_vol import garch_smile
from delta_hedge import (rebalance_frequency_comparison,
                          vol_mismatch_analysis)
from performance import (plot_greeks, plot_binomial_convergence,
                          plot_mc_paths, plot_vol_surface,
                          plot_vol_comparison, plot_endogenous_smile,
                          plot_delta_hedge)

print("=" * 60)
print("VN30 Options Pricing Engine")
print("=" * 60)

T = 0.25   # 3-month option

# ── 1. Load market data + fit GARCH ─────────────────────────────────────────────
print("\nStep 1: Loading market data and fitting GARCH(1,1)...")
inputs       = get_model_inputs(ticker="VCB", vol_window=30, T=T)
S            = inputs["S"]
r            = inputs["r"]
sigma_hist   = inputs["sigma"]           # backward-looking flat vol
sigma_garch  = inputs["sigma_garch"]     # GARCH Level-1 forecast σ
garch_params = inputs["garch_params"]    # ω, α, β, σ²_t for Level 2
K            = S                          # ATM

print(f"\nPricing: ATM {T*12:.0f}-month option on VCB")
print(f"  S={S:,.0f} VND  K={K:,.0f} VND  r={r*100:.1f}%")
print(f"  σ_hist  = {sigma_hist*100:.2f}%  "
      f"(last {30}d rolling stdev of log returns)")
print(f"  σ_garch = {sigma_garch*100:.2f}%  "
      f"(GARCH Level-1 forecast over [0, T])")

# ── 2. Black-Scholes under both vol inputs ──────────────────────────────────────
print("\nStep 2: Black-Scholes pricing under both vol inputs...")
C_hist  = call_price(S, K, T, r, sigma_hist)
P_hist  = put_price (S, K, T, r, sigma_hist)
C_garch = call_price(S, K, T, r, sigma_garch)
P_garch = put_price (S, K, T, r, sigma_garch)

print(f"  Historical σ → Call = {C_hist:,.0f}  Put = {P_hist:,.0f}")
print(f"  GARCH-L1  σ → Call = {C_garch:,.0f}  Put = {P_garch:,.0f}")

pcp = put_call_parity_check(S, K, T, r, sigma_garch)
print(f"  Put-call parity error: {pcp['Parity error']:.2e}")

call_greeks = all_greeks(S, K, T, r, sigma_garch, "call")
print(f"  Call Greeks (σ_garch): "
      f"Δ={call_greeks['Delta']:.3f}  "
      f"Γ={call_greeks['Gamma']:.2e}  "
      f"ν={call_greeks['Vega']:,.0f}  "
      f"Θ={call_greeks['Theta']:,.0f}")

# ── 3. Binomial tree ────────────────────────────────────────────────────────────
print("\nStep 3: Binomial tree pricing (σ_garch)...")
bin_call = european_price(S, K, T, r, sigma_garch, N=500, option_type="call")
bin_put  = european_price(S, K, T, r, sigma_garch, N=500, option_type="put")
am_put   = american_price(S, K, T, r, sigma_garch, N=500, option_type="put")
eep      = early_exercise_premium(S, K, T, r, sigma_garch)

print(f"  European Call = {bin_call:,.0f} VND (BS diff: {bin_call-C_garch:+.2f})")
print(f"  European Put  = {bin_put:,.0f} VND (BS diff: {bin_put-P_garch:+.2f})")
print(f"  American Put  = {am_put:,.0f} VND")
print(f"  Early exercise premium = {eep['Early Exercise Premium']:,.0f} VND "
      f"({eep['Premium as % of Euro']:.2f}% of European)")

# ── 4. Monte Carlo: three variance-reduction methods + Duan GARCH ───────────────
print("\nStep 4: Monte Carlo pricing (σ_garch, 10,000 paths)...")
mc_naive_res = mc_naive          (S, K, T, r, sigma_garch,
                                   n_paths=10000, option_type="call")
mc_anti_res  = mc_antithetic     (S, K, T, r, sigma_garch,
                                   n_paths=10000, option_type="call")
mc_ctrl_res  = mc_control_variate(S, K, T, r, sigma_garch,
                                   n_paths=10000, option_type="call")

print(f"  Naive MC   : {mc_naive_res['price']:,.0f} VND "
      f"± {mc_naive_res['std_error']:,.0f}")
print(f"  Antithetic : {mc_anti_res ['price']:,.0f} VND "
      f"± {mc_anti_res ['std_error']:,.0f}")
print(f"  Control CV : {mc_ctrl_res ['price']:,.0f} VND "
      f"± {mc_ctrl_res ['std_error']:,.0f}")
print(f"  BS (σ_g)   : {C_garch:,.0f} VND")

print("\nStep 4b: Duan GARCH Monte Carlo (Level 2: full GARCH path)...")
mc_duan = mc_garch(S, K, T, r, garch_params,
                   n_paths=30000, option_type="call")
print(f"  Duan MC Call (full GARCH path): "
      f"{mc_duan['price']:,.0f} VND ± {mc_duan['std_error']:,.0f}")
print(f"  Average realised σ across paths: "
      f"{mc_duan['avg_realised_vol']*100:.2f}%")
print(f"  Δ vs BS(σ_garch) = {mc_duan['price'] - C_garch:+,.0f} VND  "
      f"(non-zero because BS is flat-vol; Duan captures clustering)")

# ── 5. Endogenous smile from GARCH MC ───────────────────────────────────────────
print("\nStep 5: Backing out implied-vol smile from Duan MC prices...")
smile = garch_smile(S, r, T, garch_params, n_paths=30000)
print(smile[["moneyness", "garch_mc_price", "implied_vol"]].to_string(index=False))

# ── 6. Delta hedging ────────────────────────────────────────────────────────────
print("\nStep 6: Delta hedging simulation (σ_garch, 500 paths)...")
freq_df = rebalance_frequency_comparison(S, K, T, r, sigma_garch, n_paths=500)
print(freq_df[["Frequency", "Mean P&L (no cost)",
               "Mean P&L (w/ cost)", "Cost drag"]].to_string(index=False))

vm_df = vol_mismatch_analysis(S, K, T, r,
                               implied_vol=sigma_garch,
                               realised_vols=[sigma_garch*x
                               for x in [0.5, 0.75, 1.0, 1.25, 1.5]],
                               n_paths=500)
print(f"\n  Vol mismatch: realised < implied (σ_garch) → profitable short vol")
print(vm_df[["Realised Vol (%)", "Vol Spread (%)",
             "Mean P&L", "% Profitable"]].to_string(index=False))

# ── 7. Save summary ─────────────────────────────────────────────────────────────
print("\nStep 7: Saving summary...")
os.makedirs("output", exist_ok=True)

summary = pd.DataFrame([{
    "Ticker"              : "VCB",
    "Spot (VND)"          : round(S, 0),
    "Strike (VND)"        : round(K, 0),
    "Maturity (years)"    : T,
    "Risk-free rate"      : r,
    "σ historical (30d)"  : round(sigma_hist, 4),
    "σ GARCH L1"          : round(sigma_garch, 4),
    "σ GARCH long-run"    : round(garch_params["long_run_vol_annual"], 4),
    "GARCH α+β"           : round(garch_params["persistence"], 4),
    "BS Call (σ_hist)"    : round(C_hist, 0),
    "BS Call (σ_garch)"   : round(C_garch, 0),
    "BS Put (σ_hist)"     : round(P_hist, 0),
    "BS Put (σ_garch)"    : round(P_garch, 0),
    "Binomial Call"       : round(bin_call, 0),
    "American Put"        : round(am_put, 0),
    "EEP"                 : round(eep["Early Exercise Premium"], 0),
    "MC Naive Call"       : round(mc_naive_res["price"], 0),
    "MC Antithetic"       : round(mc_anti_res ["price"], 0),
    "MC Control"          : round(mc_ctrl_res ["price"], 0),
    "Duan MC Call"        : round(mc_duan     ["price"], 0),
    "Duan MC SE"          : round(mc_duan     ["std_error"], 0),
    "Call Delta"          : call_greeks["Delta"],
    "Call Gamma"          : call_greeks["Gamma"],
    "Call Vega"           : call_greeks["Vega"],
    "Call Theta"          : call_greeks["Theta"],
}])
summary.to_csv("output/pricing_summary.csv", index=False)
smile.to_csv  ("output/endogenous_smile.csv", index=False)
print("  Saved pricing_summary.csv and endogenous_smile.csv")

# ── 8. Generate all charts ──────────────────────────────────────────────────────
print("\nStep 8: Generating charts...")
plot_greeks             (S, r, sigma_garch, "call")
plot_binomial_convergence(S, K, T, r, sigma_garch)
plot_mc_paths           (S, r, sigma_garch, T, n_paths=100)
plot_vol_surface        (S, r, sigma_garch)
plot_vol_comparison     (inputs["returns"], garch_params,
                         sigma_hist=sigma_hist, T=T)
plot_endogenous_smile   (S, r, T, garch_params,
                         sigma_hist=sigma_hist, sigma_garch=sigma_garch)
plot_delta_hedge        (S, K, T, r, sigma_garch)

print("\nDone! All outputs saved to output/")
