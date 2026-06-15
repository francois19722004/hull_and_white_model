# Hull & White Stochastic Volatility Model — Project Structure & Results

## Model Overview

This project implements and reproduces the numerical experiments from Hull & White (1987) on option pricing under stochastic volatility. The model assumes:

- The stock price follows: `dS = μS dt + √V · S dW₁`
- The variance follows a geometric Brownian motion: `dV = ξV dW₂`
- The two Brownian motions are correlated: `d<W₁, W₂> = ρ dt`

The key parameters are:
- `σ₀` (or `sigma0`): initial volatility (`V₀ = σ₀²`)
- `ξ` (xi): volatility of volatility
- `ρ` (rho): correlation between stock returns and volatility
- `μ`: drift of variance (set to 0 in most experiments)
- `T`: option maturity
- `r`: risk-free rate

The **true price** is approximated by two methods:
1. **Series solution (Eq. 9)**: analytical expansion around the B-S price up to second order in variance and skewness corrections
2. **Monte Carlo**: simulates variance paths (and joint S/V paths when ρ ≠ 0), using antithetic variates for variance reduction

---

## Project Structure

```
hull_and_white_model/
│
├── src/                             # Production scripts (clean reproductions)
│   ├── comparaison_serie_monte_carlo.py   → Table I
│   ├── biais_en_fonction_de_xhi.py        → Figure 1 (B-S vs Eq.9, bias ×25)
│   ├── biais_en_fonction_volatilite.py    → Figure 2 (bias vs σ₀)
│   ├── biais_en_fonction_de_xi.py         → Figure 3 (bias vs ξ)
│   ├── biais_en_fonction_rho.py           → Table II as a graph (bias vs ρ)
│   ├── price_bias_vs_maturity.py          → Figure 4 (bias vs maturity T)
│   ├── table_II.py                        → Table II (numeric)
│   ├── tableIII.py                        → Table III (implied vols)
│   ├── smile_de_volatilite.py             → Volatility smile figures
│   └── experiences_a_mener.txt            → Experiment planning notes
│
├── test_experiments/                # Exploratory/experimental scripts
│   ├── exp1_jensen_effect.py
│   ├── exp2_validation_xi0.py
│   ├── exp3_table_I.py
│   ├── exp4_figure1.py
│   ├── exp5_sensitivity.py
│   ├── exp6_convergence.py
│   ├── exp7_table_II.py
│   ├── exp8_smiles.py
│   ├── exp9_table_III.py
│   └── exp10_figure4_maturity.py
│
├── experiments_in_the_article/      # Screenshots of the original paper figures
│   └── Screenshot 2026-06-14 *.png
│
└── src/<script_name>/               # Each script auto-saves its output here
    ├── table_I.txt
    ├── figure_1.png  (biais_en_fonction_de_xhi/)
    ├── figure_2.png  (biais_en_fonction_volatilite/)
    ├── figure_3.png  (biais_en_fonction_de_xi/)
    ├── figure_4.png  (price_bias_vs_maturity/)
    ├── biais_vs_rho.png
    ├── table_II.txt
    ├── table_III.txt
    └── smile_*.png   (smile_de_volatilite/)
```

---

## Experiments & Results

### Table I — Monte Carlo vs Series Solution (`comparaison_serie_monte_carlo.py`)

**Parameters:** σ₀ = 10%, ξ = 1, μ = 0, T = 180 days, r = 0, ρ = 0

Compares three prices across a range of moneyness S/X ∈ [0.75, 1.25]:
- Black-Scholes price
- Series solution (Eq. 9): two-term correction for variance and skewness
- Monte Carlo price (10,000 simulations, 180 steps)

**Result:** The series solution and MC prices are very close to the B-S price near ATM; the bias is small (~1–3%) and the series captures it well. Validates that ρ = 0 and low ξ keep B-S approximately correct.

**Output:** `src/comparaison_serie_monte_carlo/table_I.txt`

---

### Figure 1 — Pricing Bias, B-S vs True Price (`biais_en_fonction_de_xhi.py`)

**Parameters:** σ₀ = 15%, ξ = 1, μ = 0, T = 180 days, r = 0, ρ = 0

Plots the B-S price vs the Eq. 9 price with the bias **exaggerated 25×** for visibility. Vertical lines mark the crossover points (S/X where bias changes sign).

**Result:** B-S overestimates deep OTM and deep ITM options, and underestimates near-the-money options. The bias is symmetric around a near-ATM crossing point.

**Output:** `src/biais_en_fonction_de_xhi/figure_1.png`

---

### Figure 2 — Bias vs Initial Volatility σ₀ (`biais_en_fonction_volatilite.py`)

**Parameters:** ξ = 1, μ = 0, T = 180 days, r = 0, ρ = 0; σ₀ ∈ {10%, 15%, 20%}

For each σ₀, plots the percentage bias `(P_HW − P_BS) / P_BS × 100` as a function of S/X.

**Result:** Higher σ₀ flattens the bias curve (smaller relative bias). The shape remains the same: positive bias near ATM, negative in the wings. Bias stays within ±5%.

**Output:** `src/biais_en_fonction_volatilite/figure_2.png`

---

### Figure 3 — Bias vs Volatility-of-Volatility ξ (`biais_en_fonction_de_xi.py`)

**Parameters:** σ₀ = 15%, μ = 0, T = 180 days, r = 0, ρ = 0; ξ ∈ {1, 2, 3}

For each ξ, plots the percentage bias using 50,000 MC simulations with antithetic variates.

**Result:** Higher ξ dramatically amplifies the bias (up to −25% for deep OTM with ξ = 3). This is the Jensen's inequality effect: a convex B-S function applied to a stochastic mean variance produces systematic mispricing that grows with volatility-of-volatility.

**Output:** `src/biais_en_fonction_de_xi/figure_3.png`

---

### Bias vs ρ (Table II as Graph) (`biais_en_fonction_rho.py`)

**Parameters:** σ₀ = 15%, ξ = 1, μ = 0, T = 180 days, r = 0; ρ ∈ {−1, −0.5, 0, 0.5, 1}

Uses **joint simulation** of (S, V) with antithetic variates on the stock Brownian motion.

**Result:** Correlation ρ breaks the symmetry of the bias curve. Negative ρ (volatility rises when stock falls) skews the bias toward a negative (leftward) tilt — consistent with the equity volatility skew observed in markets. Positive ρ does the opposite.

**Output:** `src/biais_en_fonction_rho/biais_vs_rho.png`

---

### Figure 4 — Bias vs Maturity (`price_bias_vs_maturity.py`)

**Parameters:** σ₀ = 15%, ξ = 1, μ = 0, **r = 10%**; T ∈ {45, 90, 135} days

**Result:** Longer maturities increase the bias magnitude, because there is more time for stochastic volatility to deviate from σ₀. The non-zero rate shifts the crossover point away from S/X = 1.

**Output:** `src/price_bias_vs_maturity/figure_4.png`

---

### Table II — Bias as % of B-S Price (`table_II.py`)

**Parameters:** σ₀ = 15%, ξ = 1, μ = 0, r = 0; S/X ∈ {0.90, 0.95, 1.00, 1.05, 1.10}; ρ ∈ {−1, −0.5, 0, 0.5, 1}; T ∈ {90, 180, 270} days

Uses joint MC (15,000 sims, 90 steps) with antithetic variates. Reports bias (%) and standard error for each (S/X, ρ, T) combination.

**Result:** A full sensitivity matrix showing how bias depends jointly on moneyness, correlation, and maturity. At ρ = 0, bias is symmetric. At ρ = ±1, the skew is most pronounced.

**Output:** `src/table_II/table_II.txt`

---

### Table III — Implied Volatility from MC Prices (`tableIII.py`)

**Parameters:** Same as Table II.

Takes the MC option prices from Table II and back-solves for implied volatility via bisection on the B-S formula. Reports σ_imp (%) and its standard error (via delta method: `se_vol = se_price / vega`).

**Result:** The volatility smile. When ρ = 0, implied vol is approximately flat (≈ 15%) across strikes. When ρ ≠ 0, a pronounced skew appears: negative ρ produces the classic "smirk" (higher implied vol for OTM puts), positive ρ produces the reverse skew.

**Output:** `src/tableIII/table_III.txt`

---

### Volatility Smiles (`smile_de_volatilite.py`)

Generates four figures showing the implied volatility smile under different parameter regimes:

| Figure | Varies | Fixed |
|--------|--------|-------|
| `smile_reference.png` | — | ρ = 0, ξ = 1, T = 180j |
| `smile_effet_xi.png` | ξ ∈ {0.5, 1, 2} | ρ = 0, T = 180j |
| `smile_effet_rho.png` | ρ ∈ {−1, −0.5, 0, 0.5, 1} | ξ = 1, T = 180j |
| `smile_effet_T.png` | T ∈ {90, 180, 270} days | ρ = 0, ξ = 1 |

Uses 50,000 MC sims with antithetic variates. Implied vol is computed via bisection.

**Results:**
- **Reference:** nearly flat smile at ~15%, confirming B-S approximation holds when ρ = 0 and ξ is moderate
- **Effect of ξ:** larger ξ creates a more pronounced U-shaped smile (symmetric, since ρ = 0)
- **Effect of ρ:** negative ρ tilts the smile into a skew/smirk; positive ρ inverts it
- **Effect of T:** longer maturities flatten the smile (stochastic vol effects average out)

**Output:** `src/smile_de_volatilite/smile_*.png`

---

## Key Implementation Notes

- **Two MC methods** are used depending on ρ:
  - `mc_rho0`: when ρ = 0, simulates only the variance path and prices via `E[BS(√V̄)]` — faster
  - `mc_joint`: when ρ ≠ 0, simulates (S, V) jointly step by step using the Cholesky decomposition `dW_V = ρ dW_S + √(1−ρ²) dW_⊥`
- **Antithetic variates** are applied by negating the stock Brownian motion (U → −U) while keeping the orthogonal noise (Vn) unchanged
- **Implied vol inversion** uses bisection on [1e-6, 5.0] with tolerance 1e-8
- All scripts save output to a subdirectory named after the script (e.g., `src/smile_de_volatilite/`)
