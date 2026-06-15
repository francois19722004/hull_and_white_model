"""
Smile de volatilite implicite — effet de rho (version amelioree)

Differences vs smile_de_volatilite.py (figure smile_effet_rho uniquement):
- S/X range: [0.90, 1.10] (11 points, pas de 0.02)
- rho=0: mc_rho0 (V-path only + analytical BS), N=100000
- rho!=0: mc_joint (joint S,V simulation), N=100000
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import time

script_name = os.path.splitext(os.path.basename(__file__))[0]
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
os.makedirs(output_dir, exist_ok=True)

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

def implied_vol_bisection(S, K, r, T, price, tol=1e-8, max_iter=300):
    lower_bound = max(S - K * np.exp(-r * T), 0.0)
    if price <= lower_bound + 1e-10 or price >= S - 1e-10:
        return np.nan
    sig_low, sig_high = 1e-6, 5.0
    for _ in range(max_iter):
        sig_mid = (sig_low + sig_high) / 2
        p = bs_call(S, K, r, T, sig_mid)
        if abs(p - price) < tol:
            return sig_mid
        if p < price:
            sig_low = sig_mid
        else:
            sig_high = sig_mid
    return sig_mid

def mc_rho0(S0, K, r, T, V0, xi, n_steps, n_sims):
    """MC vectorise, rho=0, mu=0, avec antithetiques. Simule V seulement, prix = E[C_BS(sqrt(Vbar))]."""
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

    return np.mean((P1 + P2) / 2)

def mc_joint(S0, K, r, T, V0, xi, rho, n_steps, n_sims):
    """MC vectorise, simulation conjointe S et V, antithetiques sur u."""
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_1_rho2 = np.sqrt(1 - rho**2)

    U = np.random.randn(n_sims, n_steps)
    Vn = np.random.randn(n_sims, n_steps)

    S1 = np.full(n_sims, S0)
    V1 = np.full(n_sims, V0)
    S2 = np.full(n_sims, S0)
    V2 = np.full(n_sims, V0)

    for i in range(n_steps):
        u_i = U[:, i]
        v_i = Vn[:, i]
        S1 = S1 * np.exp((r - V1/2) * dt + np.sqrt(np.maximum(V1, 0)) * sqrt_dt * u_i)
        V1 = np.maximum(V1 * np.exp((-xi**2/2) * dt + xi * sqrt_dt * (rho * u_i + sqrt_1_rho2 * v_i)), 1e-12)
        S2 = S2 * np.exp((r - V2/2) * dt + np.sqrt(np.maximum(V2, 0)) * sqrt_dt * (-u_i))
        V2 = np.maximum(V2 * np.exp((-xi**2/2) * dt + xi * sqrt_dt * (rho * (-u_i) + sqrt_1_rho2 * v_i)), 1e-12)

    p1 = np.exp(-r * T) * np.maximum(S1 - K, 0)
    p2 = np.exp(-r * T) * np.maximum(S2 - K, 0)
    return np.mean((p1 + p2) / 2)

# Parametres
S0 = 1.0
r = 0.0
sigma0 = 0.15
V0 = sigma0**2
xi = 1.0
T = 180 / 365
n_steps = 90
n_sims = 100000

moneyness = np.linspace(0.90, 1.10, 11)

rho_configs = [
    (-1.0, 'darkblue', 'v'),
    (-0.5, 'royalblue', '<'),
    (0.0,  'green',    'o'),
    (0.5,  'orange',   '>'),
    (1.0,  'red',      '^'),
]

fig, ax = plt.subplots(figsize=(10, 6))

for rho_val, color, marker in rho_configs:
    t0 = time.time()
    sigma_imps = []
    for sx in moneyness:
        K = S0 / sx
        if abs(rho_val) < 1e-10:
            price = mc_rho0(S0, K, r, T, V0, xi, n_steps, n_sims)
        else:
            price = mc_joint(S0, K, r, T, V0, xi, rho_val, n_steps, n_sims)
        sig = implied_vol_bisection(S0, K, r, T, price)
        sigma_imps.append(sig * 100 if not np.isnan(sig) else np.nan)

    valid = [not np.isnan(s) for s in sigma_imps]
    ax.plot(moneyness[valid], np.array(sigma_imps)[valid], f'{marker}-', color=color,
            linewidth=2, markersize=6, label=r'$\rho$ = ' + f'{rho_val}')
    print(f"  rho={rho_val:+.1f} termine ({time.time()-t0:.1f}s)")

ax.axhline(sigma0 * 100, color='gray', linestyle='--', alpha=0.5, label=r'$\sigma_0$ = 15%')
ax.set_xlabel('S / X', fontsize=14)
ax.set_ylabel(r'$\sigma_{imp}$ (%)', fontsize=14)
ax.set_title(r'Effet de $\rho$ sur le smile ($\xi=1$, $T=180j$)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

filepath = os.path.join(output_dir, 'smile_effet_rho.png')
plt.tight_layout()
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSauvegarde: {filepath}")
