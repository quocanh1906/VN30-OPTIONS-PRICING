# VN30 Options Pricing Engine

An options pricing and risk-management framework for Vietnamese equities, built on VCB (Vietcombank) as the most liquid VN30 constituent. It implements three pricers (Black–Scholes, binomial tree, Monte Carlo), two volatility estimators (rolling historical and GARCH(1,1)), dynamic delta hedging, and — critically — a **Duan (1995) GARCH Monte Carlo pricer** that generates an endogenous volatility smile without relying on any listed-options market data.

This project directly applies theoretical knowledge from the Imperial College, Financial Engineering and Advanced Options Theory modules to real Vietnamese market data, and connects to the structured products risk assessment work done at Techcom Securities.

The last point matters. Vietnam has no liquid listed-options market, so there is no implied-volatility surface to borrow from. Most textbook options projects start from an IV input; we cannot. Everything vol-related here is estimated from the historical return series.

---

## Table of Contents

1. [The Core Problem: No Liquid Options → No IV](#the-core-problem)
2. [Design Principle: What Each Pricer Does & Why](#design-principle)
3. [Assumptions and Where They Break](#assumptions)
4. [The Volatility-Input Hierarchy (Level 1 vs Level 2)](#vol-hierarchy)
5. [Why Duan GARCH MC Cannot Live Inside Black–Scholes](#why-duan-is-mc)
6. [Module-by-Module Walk-Through](#walkthrough)
7. [Reproducing the Results](#reproducing)
8. [Output Artifacts](#outputs)
9. [Limitations / Things a Real Desk Would Add](#limitations)

---

<a name="the-core-problem"></a>
## 1. The Core Problem: No Liquid Options → No IV

In a mature market (S&P 500, HSI, FTSE), an options pricing stack takes implied volatility from observable market prices and feeds it into a model. Calibration is the work. In Vietnam, there is no such input:

- No listed equity options.
- No volatility futures, no VIX-equivalent for the VN30.
- OTC warrants exist but are thinly quoted and heavily dealer-driven.

So the volatility number has to be estimated from historical returns. Any project that pretends to "imply" vol from synthetic prices (generate prices from an assumed smile, then recover that same smile) is solving a circular problem — the recovered vol *is* the input by construction. This project is honest about that:

- The synthetic-smile module (`build_vol_surface`) is retained as a methodology demonstration but is clearly marked as circular.
- The **real** smile this project produces comes from the GARCH Monte Carlo (§5).

---

<a name="design-principle"></a>
## 2. Design Principle: What Each Pricer Does & Why

Three pricing approaches, each earning its place:

| Pricer | What it needs | What it's good at | What it can't do |
|---|---|---|---|
| Black–Scholes (closed-form) | one scalar σ | speed; exact Greeks | anything time-varying |
| Binomial tree (CRR) | one scalar σ | American exercise; intuition | exotic payoffs; path-dependence |
| Monte Carlo | either scalar σ or a full variance process | any payoff; path-dependence; **real GARCH dynamics** | early exercise (naturally); slow convergence |

The MC pricer is the only one that can ingest the *full* GARCH process — the other two are mathematically restricted to constant σ. This is not an implementation limitation; it's a theorem. See §5.

---

<a name="assumptions"></a>
## 3. Assumptions and Where They Break

Every options model rests on assumptions that are violated in practice. Being explicit matters more than pretending otherwise.

### 3.1 Black–Scholes (1973)

| Assumption | Reality (Vietnam) | Our mitigation |
|---|---|---|
| Geometric Brownian Motion for S | Returns exhibit vol clustering, fat tails | Duan MC relaxes this |
| Constant volatility σ | Vol clusters and mean-reverts | GARCH Level-1 σ scalar; Duan MC full path |
| Constant risk-free rate r | 4.5% VND-gov-bond approximation | Held fixed, flagged |
| No dividends | VCB pays ~0.8% quarterly dividends | **Not yet modelled — documented limitation** |
| Continuous trading, no frictions | HOSE has tick sizes, intraday caps, price limits | Acknowledged; delta-hedge sim uses real VN costs |
| European exercise only | Moot (no listed market) | Binomial handles American when needed |

### 3.2 Binomial (CRR)

Same as BS for pricing, plus the extra assumption that the risk-neutral up-move and down-move are given by `u = exp(σ√Δt)`, `d = 1/u`. This is a CRR choice; Jarrow–Rudd or Tian trees give slightly different convergence. At N = 500 steps the error to BS is <0.1% in our setup.

### 3.3 Monte Carlo

- Standard MC (`mc_naive`, `mc_antithetic`, `mc_control_variate`) assumes GBM with constant σ — same as BS; variance-reduction methods change the *estimator*, not the *model*.
- Duan MC (`mc_garch`) assumes the risk-neutralised GARCH(1,1) process of Duan (1995). Key simplification: **λ = 0** (no volatility risk premium priced). This is reasonable as a first pass in a market with no listed options but would be recalibrated from traded prices in a market that has them.

### 3.4 Vietnamese market-specific

- **Transaction costs**: 0.125% commission + 0.10% sales tax on sell side. Used in the delta-hedge P&L sim.
- **Price units**: HOSE quotes in thousands of VND (60.56 = 60,560 VND). `data.py` scales on load so every downstream number is in actual VND — no unit confusion possible inside the code.

---

<a name="vol-hierarchy"></a>
## 4. The Volatility-Input Hierarchy

Three levels of vol input, in order of fidelity:

### Level 0 — Rolling historical σ  *(baseline)*

```
σ_hist = stdev(last 30 log-returns) · √252
```

One backward-looking scalar. Equal-weighted window. Ignores vol clustering completely.

**What it gets wrong**: If yesterday closed at 2% daily move and the 30 days before were 0.5%, rolling 30d σ barely budges. Everyone on the desk knows vol just spiked, but the number doesn't reflect it.

### Level 1 — GARCH(1,1) scalar forecast

```
Fit:     σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
Forecast: project σ²_{t+h} forward for h = 1..⌈T·252⌉ days
Average:  σ_{GARCH} = √(mean(σ²_{t+h}) · 252)
```

Still a single scalar — good enough for BS and the binomial tree. But this scalar *knows* about the current vol regime (via σ²_t) and the long-run level (via ω/(1−α−β)), and it decays from one to the other at rate (α+β) per day.

**What it gets right**: vol clustering, mean reversion.
**What it still gets wrong**: the terminal distribution is still log-normal by assumption in BS/binomial; the smile is still flat.

For VCB at end-2024: Level 0 → 10.1%, Level 1 → 19.2%. Level 1 is almost 2× Level 0 because the rolling 30d sample happens to sit in a calm regime while the long-run is materially higher (27%).

### Level 2 — Duan (1995) GARCH Monte Carlo

The full GARCH process is simulated path-by-path under the risk-neutral measure. Each path's variance evolves day-by-day following the fitted recursion. No collapse to a scalar.

```
For each MC path, for each trading day t:
  log-return    r_t   = r_daily - ½σ²_t + σ_t · ε*_t,      ε*_t ~ N(0,1)
  variance update  σ²_{t+1} = ω + α · σ²_t · ε*_t² + β · σ²_t
```

This is the only pricer that produces a **non-flat** smile without imposing one. See §5 for why only MC can do this, and the `output/endogenous_smile.png` panel for what the smile looks like on VCB.

### Direct comparison on VCB (ATM 3M call)

| Vol input | σ (%) | BS call (VND) | MC call (VND) |
|---|---|---|---|
| Level 0: historical 30d | 10.10 | 1,582 | — |
| Level 1: GARCH scalar | 19.17 | 2,656 | — |
| Level 2: Duan GARCH path | *endogenous* | — | 2,633 ± 22 |

The huge gap between Level 0 and Level 1 (+68%) is precisely the information the rolling-window estimator throws away. Level 1 and Level 2 agree at ATM (as they must); their difference shows up in the wings.

---

<a name="why-duan-is-mc"></a>
## 5. Why Duan GARCH MC Cannot Live Inside Black–Scholes

This comes up repeatedly, so it's worth stating the theorem clearly:

> Black–Scholes has a closed-form solution **because** σ is constant. If you allow variance to follow a path-dependent process (as GARCH does), the BS integral no longer has an analytical form. You must simulate paths.

More concretely:

- BS price = `S·N(d1) − K·e^(−rT)·N(d2)`, where `d1`, `d2` depend on **one** σ.
- Duan GARCH dynamics → terminal `S_T` distribution is *not* log-normal (it has extra kurtosis and possibly skew from the variance clustering).
- `N(d1)` is a cdf of a Gaussian. It cannot represent that non-Gaussian distribution.

So the MC pricer *is* the Level-2 pricer. There is no separate "BS-with-Duan" version to implement — the MC engine that evolves the GARCH process is exactly that object.

**Corollary**: if you back out BS-equivalent IV from Duan MC prices (strike by strike), you get a smile. That smile is the BS cdf's "error bar" against a non-log-normal truth. It is a genuine empirical output — not an assumed parameter.

This is what makes `src/implied_vol.py:garch_smile()` non-circular for the first time in this project.

---

<a name="walkthrough"></a>
## 6. Module-by-Module Walk-Through

### `src/data.py`

Loads VCB daily close prices from the sibling VN30-MA-Crossover project (no re-download), computes log returns, rolling historical vol at 30/60/252d windows, and — if `fit_garch_model=True` — also fits a GARCH(1,1) model and returns its Level-1 scalar forecast plus the full param dict for Level-2 use.

HOSE quotes in thousands of VND; `data.py` multiplies by `PRICE_UNIT = 1000` at load so downstream code sees real VND everywhere.

### `src/garch_vol.py` *(new)*

Thin wrapper around the `arch` library. Three functions:

- `fit_garch(returns)` → ω, α, β, persistence, long-run vol, current σ²_t.
- `garch_sigma_for_horizon(params, T)` → Level-1 scalar for BS/binomial.
- `in_sample_conditional_vol(params)` → σ_t series for diagnostic plots.

The project uses zero-mean GARCH(1,1) with Gaussian innovations. This is the textbook default; Student-t innovations would be a natural extension.

### `src/black_scholes.py`

Closed-form call, put, put-call parity check, and all five Greeks (Δ, Γ, ν, Θ, ρ). No changes to the math — the module is as theoretically clean as BS gets.

### `src/binomial.py`

Cox-Ross-Rubinstein tree with `u = exp(σ√Δt)`, `d = 1/u`. Prices European (backward induction) and American (early-exercise check at every node). Converges to BS for N ≥ 200 steps.

### `src/monte_carlo.py`

Four pricers:

| Function | Model | Variance reduction |
|---|---|---|
| `mc_naive` | GBM, const σ | none |
| `mc_antithetic` | GBM, const σ | mirror paths |
| `mc_control_variate` | GBM, const σ | control = `S_T` (not the payoff!) |
| `mc_garch` | **Duan 1995 GARCH(1,1)** | none by default |

The control-variate fix matters: using the payoff itself as the control makes β → 1 and collapses the estimator to `bs_true` with zero standard error — a silent bug. The correct control is the terminal stock price `S_T`, which has known risk-neutral expectation `S·e^(rT)`.

### `src/implied_vol.py`

- `implied_vol(price, S, K, T, r, type)`: Brent root-find; correctly bounded for both calls (`price ≤ S`) and puts (`price ≤ K·e^(−rT)`).
- `compute_smile / build_vol_surface`: **synthetic** (skew, smile_curve imposed; retained as demonstration).
- `garch_smile`: **real** — prices each strike via Duan MC, inverts BS on the price, returns BS-equivalent IV curve.

### `src/delta_hedge.py`

Simulates dealer-side short call + dynamic delta hedge. Uses Vietnamese transaction-cost parameters. Produces:

- P&L distribution across N paths.
- Rebalance-frequency comparison (daily / weekly / monthly; gamma risk vs cost trade-off).
- Vol-mismatch scenario: implied vs realised vol → confirms the classic "short vol profits when realised < implied" result.

### `src/performance.py`

All visualisations. Two new charts added:

- `plot_vol_comparison`: time-series of rolling vs GARCH conditional σ; forward term structure; return-distribution diagnostic (fat tails vs Gaussian).
- `plot_endogenous_smile`: the money chart — flat historical / flat GARCH-scalar lines vs. the curved GARCH MC smile.

---

<a name="reproducing"></a>
## 7. Reproducing the Results

```bash
cd VN30-Options-Pricing
python main.py
```

Expected runtime: ~90s on a laptop (dominated by 30k-path MC runs across 9 strikes for the smile).

Requirements: `numpy pandas scipy matplotlib arch`. The `arch` package is the only non-standard dependency (used for GARCH fitting).

Each module is also runnable standalone for inspection:

```bash
python src/garch_vol.py          # fit GARCH on VCB, show param + σ forecasts
python src/monte_carlo.py        # compare naive/antithetic/control
python src/delta_hedge.py        # single-path hedge trace + frequency sweep
```

---

<a name="outputs"></a>
## 8. Output Artifacts

| File | What it shows |
|---|---|
| `output/pricing_summary.csv` | All ATM prices side-by-side under σ_hist and σ_garch |
| `output/endogenous_smile.csv` | Moneyness × MC price × BS-inverted IV |
| `output/greeks.png` | Δ, Γ, ν, Θ, ρ as functions of spot |
| `output/binomial_convergence.png` | Price and error vs tree-size N |
| `output/mc_paths.png` | Simulated GBM fans + terminal-price histogram |
| `output/vol_surface.png` | Synthetic smile surface (methodology demo) |
| `output/vol_comparison.png` | **Historical vs GARCH — 3-panel vol diagnostic** |
| `output/endogenous_smile.png` | **Flat-vol assumption vs curved GARCH smile** |
| `output/delta_hedge.png` | Hedge P&L distribution / freq trade-off / vol mismatch |

---

<a name="limitations"></a>
## 9. Limitations / Things a Real Desk Would Add

Honest accounting of what this project does *not* yet do:

1. **Dividends** — VCB pays quarterly dividends (~0.8% each); BS with `q = 0` biases calls up and puts down by ~3% cumulative for a 3-month option. Easy to add as `q` parameter throughout.
2. **Stochastic rates** — r is held constant at 4.5%. For options >1 year this matters; for 3-month calls it doesn't move prices much.
3. **Jump risk** — Neither GBM nor GARCH(1,1) captures discrete jumps (earnings, policy announcements). A Merton jump-diffusion or Bates model would price tail risk better; the machinery to add it is already in the MC engine.
4. **Student-t / non-Gaussian GARCH innovations** — Empirical Vietnamese returns have kurtosis > 3; GARCH with Gaussian shocks underestimates tails. `arch` supports `dist='t'`.
5. **Duan's λ (volatility risk premium)** — Set to 0. In a market with listed options, λ would be calibrated to traded prices. Here there's nothing to calibrate against.
6. **Longstaff–Schwartz** for American MC — currently we rely on the binomial tree for American pricing. For higher-dimensional problems (basket / multi-asset), LSM in the MC engine would be the right extension.
7. **No liquid VN30 option benchmark** — There's literally no market price to validate against. The best cross-check we can do is intra-method: BS ↔ binomial ↔ Duan-MC-at-ATM should all agree, and they do.

None of these are hard to add once there's an actual trading use-case demanding them. The project is structured so each module sits behind a clean function boundary — drop-in replacements are straightforward.

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.*
- Bollerslev, T. (1986). *Generalised Autoregressive Conditional Heteroscedasticity.*
- Duan, J.-C. (1995). *The GARCH Option Pricing Model.*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering* — Ch. 4 (variance reduction).
