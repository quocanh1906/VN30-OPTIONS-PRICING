# VN30 Options Pricing Engine

A complete options pricing and risk management framework built on VCB (Vietcombank) — the most liquid stock in the VN30 index. Implements the three industry-standard pricing approaches (Black-Scholes, Binomial Tree, Monte Carlo), implied volatility surface construction, and dynamic delta hedging simulation.

This project directly applies theoretical knowledge from the Imperial College, Financial Engineering and Advanced Options Theory modules to real Vietnamese market data, and connects to the structured products risk assessment work done at Techcom Securities.

---

## Key Results

### Pricing Comparison — ATM 3-Month Call on VCB
*(S = 60.6, K = 60.6, T = 0.25y, r = 4.5%, σ = 10.10%)*

| Method | Price | Notes |
|---|---|---|
| Black-Scholes | 1.5821 | Analytical closed-form |
| Binomial (N=500) | 1.5815 | Error: 0.0006 (0.04%) |
| Monte Carlo Naive | 1.5688 ± 0.0204 | 10,000 paths |
| Monte Carlo Antithetic | 1.5777 ± 0.0134 | 34% variance reduction |
| Monte Carlo Control CV | 1.5821 ± 0.0000 | Matches BS exactly |

Put-call parity error: 0.00e+00 — confirms model consistency.

---

### Greeks (ATM Call, 3M)

| Greek | Value | Interpretation |
|---|---|---|
| Delta (Δ) | 0.598 | Option gains 0.598 for every 1 VND move in VCB |
| Gamma (Γ) | 0.127 | Delta changes by 0.127 per 1 VND move |
| Vega (ν) | 11.714 | Option gains 11.714 for 1% increase in vol |
| Theta (Θ) | -0.0108 | Option loses 0.0108 per calendar day |

---

### Binomial Tree — American vs European Put

| | Price |
|---|---|
| European Put | 0.9040 |
| American Put | 0.9677 |
| Early Exercise Premium | 0.0642 (7.11% of European) |

The American put commands a 7.11% premium over the European — rational early exercise has material value even for a 3-month option at current Vietnamese rates.

---

### Monte Carlo Convergence

Demonstrates the 1/√N convergence rate — to halve the error, you need 4× more paths:

| N Paths | Naive SE | Antithetic SE | Control SE |
|---|---|---|---|
| 100 | 0.1958 | 0.1457 | 0.0 |
| 1,000 | 0.0656 | 0.0407 | 0.0 |
| 10,000 | 0.0204 | 0.0134 | 0.0 |
| 50,000 | 0.0092 | 0.0060 | 0.0 |

Asian option prices at 45% discount to European — averaging dampens the effective volatility significantly, relevant for commodity-linked structured products.

---

### Implied Volatility Surface

Synthetic surface constructed with:
- ATM vol: 10.10% (30-day historical from VCB)
- Skew: -10% (negative equity skew — OTM puts command premium)
- Smile curvature: +5%

The surface shows the classic **put skew** — OTM puts at 80% ATM carry ~20% implied vol vs 10% for ATM options at 3-month maturity. The skew flattens significantly at longer maturities (2-year surface is nearly flat), consistent with mean reversion reducing the probability of extreme moves over longer horizons.

**Key finding:** The smile directly contradicts Black-Scholes' constant volatility assumption. In practice, options desks maintain a vol surface and interpolate from it rather than using a single σ input to BS.

---

### Delta Hedging Simulation

**Rebalancing frequency tradeoff** (500 paths, short ATM call):

| Frequency | Mean P&L (no cost) | Mean P&L (w/ cost) | Cost Drag |
|---|---|---|---|
| Daily | +0.0047 | -0.3888 | 0.3935 |
| Weekly | +0.0077 | -0.2454 | 0.2531 |
| Monthly | +0.0516 | -0.1526 | 0.2042 |

For VCB at σ=10.1%, **transaction costs dominate over gamma risk**. Less frequent rebalancing is optimal — the cost savings outweigh the increase in hedging error. This finding reverses for high-volatility stocks where gamma losses would dominate.

**Volatility mismatch — the core of vol trading:**

| Realised Vol | Implied Vol | Vol Spread | Mean P&L | % Profitable |
|---|---|---|---|---|
| 5.1% | 10.1% | +5.1% | +0.5786 | 100.0% |
| 7.6% | 10.1% | +2.5% | +0.2971 | 99.6% |
| 10.1% | 10.1% | 0.0% | +0.0047 | 49.4% |
| 12.6% | 10.1% | -2.5% | -0.2930 | 3.2% |
| 15.2% | 10.1% | -5.1% | -0.5942 | 0.2% |

When realised vol equals implied vol, the strategy is approximately break-even (49.4% profitable). A dealer who sells options at 10.1% implied vol and realised vol turns out to be 15.2% will lose nearly every time. This is the fundamental risk in options dealing — vol forecasting accuracy determines profitability, not directional market views.

---

## Theoretical Framework

### Black-Scholes (1973)
Closed-form solution for European options under five assumptions:
- Geometric Brownian Motion price dynamics
- Constant volatility σ
- Constant risk-free rate r
- No dividends or transaction costs
- Continuous trading

```
Call = S × N(d1) - K × e^(-rT) × N(d2)
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

Despite its simplifying assumptions — particularly constant volatility which the vol smile directly contradicts — BS remains the industry standard quoting convention. All options are quoted in "implied vol" terms using the BS formula as the translation layer.

### Cox-Ross-Rubinstein Binomial Tree
Discrete-time model that converges to Black-Scholes as N→∞. Handles American options naturally through backward induction with early exercise check at each node. With N=500 steps, error vs BS is less than 0.04%.

### Monte Carlo Simulation
Simulates thousands of GBM paths and averages discounted payoffs. The most flexible method — naturally extends to path-dependent payoffs (Asian, barrier, lookback) that have no analytical solution.

Three variance reduction techniques implemented:
- **Antithetic variates**: pairs each path with its mirror, reduces variance ~34%
- **Control variates**: exploits correlation with BS analytical price, nearly eliminates variance for near-ATM options

### Implied Volatility
The volatility σ* that makes BS(σ*) = observed market price. Extracted numerically using Brent's root-finding method. The resulting smile/skew surface shows the market's departure from BS assumptions — particularly the higher probability assigned to downside tail events (crash risk) than the normal distribution implies.

### Delta Hedging
Dynamic replication of a sold option by trading the underlying. In continuous time, the hedge is perfect. In discrete time, hedging error arises from:
- **Gamma risk**: delta changes between rebalances, larger moves create larger errors
- **Transaction costs**: every rebalance incurs commission and sales tax

P&L of delta-hedged short option ≈ ½ × Γ × (ΔS)² - Θ × Δt

Gamma term: lose when stock moves (short gamma exposure)
Theta term: gain from time decay (short option → long theta)

---

## Data

**Underlying**: VCB (Vietcombank) daily close prices
- Source: vnstock (KBS source), reused from VN30-MA-Crossover project
- Period: 2015–2024 (2,376 daily observations)
- Price unit: thousands of VND (HOSE convention)

**Volatility inputs**:
- 30-day historical: 10.10% (primary input)
- 60-day historical: 10.16%
- 252-day historical: 16.04%
- Term structure is downward sloping — short-term vol elevated vs long-term

**Risk-free rate**: 4.5% (approximate Vietnamese 1-year government bond yield, State Bank of Vietnam, 2024)

**Implied volatility surface**: Synthetic — generated using a parametric smile model with realistic skew (-10%) and curvature (+5%) applied to the historical ATM vol. Vietnamese equity options market is nascent with limited liquidity, making real market IV extraction impractical. The methodology is real; the surface is illustrative.

---

## Assumptions and Limitations

### Assumptions
- GBM dynamics for underlying (log-normal returns)
- Constant volatility within each pricing call (not across strikes — the surface addresses this)
- Vietnamese 1-year government bond yield as risk-free rate proxy
- No dividends (VCB does pay dividends — minor pricing error)
- Prices in thousands of VND (HOSE standard convention)

### Limitations
- **No real options market data**: Vietnamese equity options market is nascent. The vol surface is synthetic and for illustration only
- **Constant volatility in BS**: contradicted by the smile — in practice, desks use local vol or stochastic vol models (Heston) for accurate surface-consistent pricing
- **American options**: MC implementation does not support American exercise (requires Longstaff-Schwartz LSM)
- **Single underlying**: full portfolio Greeks and cross-asset correlations not implemented
- **No jumps**: GBM assumes continuous paths — jump-diffusion (Merton) would better capture crash risk visible in the skew

---

## Project Structure

```
VN30-Options-Pricing/
├── main.py                ← runs full pipeline, all 5 modules
├── src/
│   ├── data.py            ← load VCB prices, compute vol inputs
│   ├── black_scholes.py   ← BS pricing, all 5 Greeks, price surface
│   ├── binomial.py        ← CRR tree, American options, convergence
│   ├── monte_carlo.py     ← naive MC, antithetic, control variates, Asian
│   ├── implied_vol.py     ← IV extraction, smile, vol surface
│   ├── delta_hedge.py     ← hedging simulation, frequency comparison, vol mismatch
│   └── performance.py     ← all charts and visualisation
├── data/
│   └── processed/         ← vol series (generated, not tracked)
└── output/
    ├── greeks.png
    ├── binomial_convergence.png
    ├── mc_paths.png
    ├── vol_surface.png
    ├── delta_hedge.png
    └── pricing_summary.csv
```

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib scipy vnstock

# Run full pipeline (uses existing VCB price data)
python main.py
```

> **Note:** Requires VCB price data from the VN30-MA-Crossover project.
> Run `python src/data.py` in that project first if data is not available.

---

## Connection to Other Projects

| Project | Focus | Connection |
|---|---|---|
| [VN30-Momentum](https://github.com/quocanh1906/VN30-Momentum) | Equity momentum | Underlying data source |
| [VN30-MA-Crossover](https://github.com/quocanh1906/VN30-MA-Crossover) | Event-driven execution | Provides daily price data |
| [VN30-Market-Risk](https://github.com/quocanh1906/VN30-Market-Risk) | VaR, GARCH, stress testing | Complements options Greeks with portfolio-level risk |
| **VN30-Options-Pricing** | Derivatives pricing | Prices the instruments the risk projects hedge |

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.
- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.* Journal of Financial Economics, 7(3), 229–263.
- Hull, J. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.* Springer.

---

## Author

Vu Quoc Anh Nguyen — MSc Risk Management & Financial Engineering, Imperial College London
GitHub: [quocanh1906](https://github.com/quocanh1906)
