"""
Analyse de convergence Monte Carlo dans le modele de Hull & White
Reutilise les fonctions BS / serie Eq.9 / MC (rho=0) deja presentes dans
comparaison_serie_monte_carlo.py et smile_de_volatilite.py.

Figure 1 : erreur standard vs N (avec / sans antithetiques) + reference 1/sqrt(N)
Figure 2 : prix MC vs nombre de pas n (N fixe, antithetiques) + reference Eq.9 / BS
Table     : facteur de reduction de variance grace aux antithetiques
Parametres : S0=1, r=0, sigma0=10%, xi=1, mu=0, rho=0, T=180j, X=S0 (ATM)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import time

np.random.seed(42)

script_name = os.path.splitext(os.path.basename(__file__))[0]
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------
# Fonctions BS / serie Eq.9 (reprises de comparaison_serie_monte_carlo.py)
# -----------------------------------------------------------------

def bs_call(S, K, r, T, sigma):
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_call_vec(S, K, r, T, sigma_vec):
    sigma_vec = np.maximum(sigma_vec, 1e-12)
    d1 = (np.log(S / K) + (r + sigma_vec**2 / 2) * T) / (sigma_vec * np.sqrt(T))
    d2 = d1 - sigma_vec * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def series_eq9(S, K, r, T, sigma, xi):
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    k = xi**2 * T
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nprime = norm.pdf(d1)
    C0 = bs_call(S, K, r, T, sigma)
    if k < 1e-10:
        return C0
    var_factor = 2.0 * (np.exp(k) - k - 1.0) / k**2 - 1.0
    term1 = 0.5 * (S * np.sqrt(T) * nprime * (d1*d2 - 1) / (4 * sigma**3)) * sigma**4 * var_factor
    skew_factor = (np.exp(3*k) - (9+18*k)*np.exp(k) + (8+24*k+18*k**2+6*k**3)) / (3*k**3)
    term2 = (1.0/6.0) * (S * np.sqrt(T) * nprime * ((d1*d2-3)*(d1*d2-1) - (d1**2+d2**2)) / (8*sigma**5)) * sigma**6 * skew_factor
    return C0 + term1 + term2

# -----------------------------------------------------------------
# Fonctions MC rho=0 (reprises de smile_de_volatilite.py)
# -----------------------------------------------------------------

def mc_rho0_antithetic(S0, K, r, T, V0, xi, n_steps, n_sims):
    """MC vectorise, rho=0, mu=0, AVEC antithetiques."""
    dt = T / n_steps
    Z = np.random.randn(n_sims, n_steps)

    log_inc = (-xi**2 / 2) * dt + xi * np.sqrt(dt) * Z
    log_V = np.zeros((n_sims, n_steps + 1))
    log_V[:, 0] = np.log(V0)
    for i in range(n_steps):
        log_V[:, i+1] = log_V[:, i] + log_inc[:, i]
    V_bar1 = np.mean(np.exp(log_V), axis=1)
    P1 = bs_call_vec(S0, K, r, T, np.sqrt(V_bar1))

    log_inc_anti = (-xi**2 / 2) * dt + xi * np.sqrt(dt) * (-Z)
    log_V_anti = np.zeros((n_sims, n_steps + 1))
    log_V_anti[:, 0] = np.log(V0)
    for i in range(n_steps):
        log_V_anti[:, i+1] = log_V_anti[:, i] + log_inc_anti[:, i]
    V_bar2 = np.mean(np.exp(log_V_anti), axis=1)
    P2 = bs_call_vec(S0, K, r, T, np.sqrt(V_bar2))

    payoffs = (P1 + P2) / 2
    return np.mean(payoffs), np.std(payoffs) / np.sqrt(n_sims)

def mc_rho0_standard(S0, K, r, T, V0, xi, n_steps, n_sims):
    """MC vectorise, rho=0, mu=0, SANS antithetiques."""
    dt = T / n_steps
    Z = np.random.randn(n_sims, n_steps)

    log_inc = (-xi**2 / 2) * dt + xi * np.sqrt(dt) * Z
    log_V = np.zeros((n_sims, n_steps + 1))
    log_V[:, 0] = np.log(V0)
    for i in range(n_steps):
        log_V[:, i+1] = log_V[:, i] + log_inc[:, i]
    V_bar = np.mean(np.exp(log_V), axis=1)
    payoffs = bs_call_vec(S0, K, r, T, np.sqrt(V_bar))

    return np.mean(payoffs), np.std(payoffs) / np.sqrt(n_sims)

# -----------------------------------------------------------------
# Parametres communs
# -----------------------------------------------------------------
S0 = 1.0
r = 0.0
sigma0 = 0.10
V0 = sigma0**2
xi = 1.0
mu = 0.0
rho = 0.0
T = 180 / 365
K = S0  # option ATM

# ===================================================================
# FIGURE 1 : Convergence en N (nombre de simulations), n fixe = 180
# ===================================================================
print("=" * 70)
print("FIGURE 1 : Convergence en N")
print("=" * 70)

n_steps_fixed = 180
N_values = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

se_anti = []
se_std = []
ci95_anti = []
ci95_std = []

for N in N_values:
    t0 = time.time()
    mean_anti, se_a = mc_rho0_antithetic(S0, K, r, T, V0, xi, n_steps_fixed, N)
    mean_std, se_s = mc_rho0_standard(S0, K, r, T, V0, xi, n_steps_fixed, N)
    se_anti.append(se_a)
    se_std.append(se_s)
    ci95_anti.append(1.96 * se_a)
    ci95_std.append(1.96 * se_s)
    print(f"  N={N:>7}  SE_std={se_s:.6f}  SE_anti={se_a:.6f}  "
          f"({time.time()-t0:.1f}s)")

se_anti = np.array(se_anti)
se_std = np.array(se_std)
N_values = np.array(N_values)

# Reference 1/sqrt(N), calee sur le premier point de la methode standard
C_ref = se_std[0] * np.sqrt(N_values[0])
ref_line = C_ref / np.sqrt(N_values)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(N_values, se_std, 'o-', color='red', linewidth=2, markersize=6,
          label='MC standard')
ax.loglog(N_values, se_anti, 's-', color='blue', linewidth=2, markersize=6,
          label='MC antithétique')
ax.loglog(N_values, ref_line, 'k--', linewidth=1.5, alpha=0.7,
          label=r'1/$\sqrt{N}$ théorique')
ax.set_xlabel('Nombre de simulations N', fontsize=14)
ax.set_ylabel('Erreur standard du prix', fontsize=14)
ax.set_title("Convergence de l'erreur standard en fonction de N", fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, which='both', alpha=0.3)

filepath1 = os.path.join(output_dir, 'convergence_nsims.png')
plt.tight_layout()
plt.savefig(filepath1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSauvegarde: {filepath1}")

# ===================================================================
# FIGURE 2 : Convergence en n (nombre de pas), N fixe = 100000, antithetique
# ===================================================================
print("\n" + "=" * 70)
print("FIGURE 2 : Convergence en n")
print("=" * 70)

N_fixed = 100000
n_values = [10, 20, 50, 100, 180, 360, 720, 1000]

mc_prices = []
for n in n_values:
    t0 = time.time()
    price, se = mc_rho0_antithetic(S0, K, r, T, V0, xi, n, N_fixed)
    mc_prices.append(price)
    print(f"  n={n:>5}  prix MC={price:.6f}  SE={se:.6f}  ({time.time()-t0:.1f}s)")

price_bs = bs_call(S0, K, r, T, sigma0)
price_eq9 = series_eq9(S0, K, r, T, sigma0, xi)
print(f"\n  Prix B-S      = {price_bs:.6f}")
print(f"  Prix Eq.9     = {price_eq9:.6f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(n_values, mc_prices, 'o-', color='blue', linewidth=2, markersize=6,
            label='Prix MC')
ax.axhline(price_eq9, color='black', linestyle='--', linewidth=1.5,
           label='Prix série (Eq. 9)')
ax.axhline(price_bs, color='gray', linestyle=':', linewidth=1.5,
           label='Prix Black-Scholes')
ax.set_xlabel('Nombre de pas de temps n', fontsize=14)
ax.set_ylabel('Prix du Call ATM', fontsize=14)
ax.set_title("Convergence du prix MC en fonction du nombre de pas n", fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

filepath2 = os.path.join(output_dir, 'convergence_nsteps.png')
plt.tight_layout()
plt.savefig(filepath2, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSauvegarde: {filepath2}")

# ===================================================================
# TABLE : Facteur de reduction de variance (antithetiques vs standard)
# ===================================================================
print("\n" + "=" * 70)
print("TABLE : Reduction de variance par antithetiques")
print("=" * 70)

N_var = 50000
n_var = 180
n_reps = 10

prices_std_reps = []
prices_anti_reps = []
se_std_reps = []
se_anti_reps = []

for rep in range(n_reps):
    t0 = time.time()
    mean_std, se_s = mc_rho0_standard(S0, K, r, T, V0, xi, n_var, N_var)
    mean_anti, se_a = mc_rho0_antithetic(S0, K, r, T, V0, xi, n_var, N_var)
    prices_std_reps.append(mean_std)
    prices_anti_reps.append(mean_anti)
    se_std_reps.append(se_s)
    se_anti_reps.append(se_a)
    print(f"  rep {rep+1:>2}/{n_reps}  prix_std={mean_std:.6f}  "
          f"prix_anti={mean_anti:.6f}  ({time.time()-t0:.1f}s)")

var_std = np.var(prices_std_reps, ddof=1)
var_anti = np.var(prices_anti_reps, ddof=1)
reduction_factor = var_std / var_anti if var_anti > 0 else np.nan

mean_se_std = np.mean(se_std_reps)
mean_se_anti = np.mean(se_anti_reps)

table_lines = []
table_lines.append("Facteur de reduction de variance par antithetiques")
table_lines.append(f"Parametres : S0={S0}, r={r}, sigma0={sigma0*100:.0f}%, xi={xi:.0f}, "
                    f"mu={mu:.0f}, rho={rho:.0f}, T=180j, K=S0 (ATM)")
table_lines.append(f"N={N_var}, n={n_var}, {n_reps} replications independantes")
table_lines.append("")
table_lines.append(f"Var(prix) standard    : {var_std:.10f}")
table_lines.append(f"Var(prix) antithetique : {var_anti:.10f}")
table_lines.append(f"Variance reduction factor (antithetic vs standard): {reduction_factor:.2f}")
table_lines.append("")
table_lines.append(f"SE standard: {mean_se_std:.6f} | SE antithetic: {mean_se_anti:.6f}")

print()
for l in table_lines:
    print(l)

filepath3 = os.path.join(output_dir, 'variance_reduction.txt')
with open(filepath3, 'w') as f:
    f.write('\n'.join(table_lines))

print(f"\nSauvegarde: {filepath3}")
