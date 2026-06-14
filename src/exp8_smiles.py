"""
Expérience 8 — Smiles de volatilité implicite
(a) Smile de référence ρ=0, ξ=1
(b) Effet de ξ : ξ = 0.5, 1, 2
(c) Effet de ρ : ρ = -1, -0.5, 0, 0.5, 1
(d) Effet de T : T = 90, 180, 270 jours
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

def bs_call(S, K, r, T, sigma):
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_call_vec(S, K, r, T, sigma_vec):
    """Prix B-S vectorisé sur un vecteur de volatilités."""
    sigma_vec = np.maximum(sigma_vec, 1e-12)
    d1 = (np.log(S / K) + (r + sigma_vec**2 / 2) * T) / (sigma_vec * np.sqrt(T))
    d2 = d1 - sigma_vec * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol_bisection(S, K, r, T, price, tol=1e-8, max_iter=200):
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

def mc_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims):
    """MC ρ=0 VECTORISÉ : simule V seul, avec antithétiques."""
    dt = T / n_steps
    Z = np.random.randn(n_sims, n_steps)
    
    log_inc = (mu - xi**2 / 2) * dt + xi * np.sqrt(dt) * Z
    log_V = np.zeros((n_sims, n_steps + 1))
    log_V[:, 0] = np.log(V0)
    for i in range(n_steps):
        log_V[:, i+1] = log_V[:, i] + log_inc[:, i]
    V_bar1 = np.mean(np.exp(log_V), axis=1)
    P1 = bs_call_vec(S0, K, r, T, np.sqrt(V_bar1))
    
    log_inc_anti = (mu - xi**2 / 2) * dt + xi * np.sqrt(dt) * (-Z)
    log_V_anti = np.zeros((n_sims, n_steps + 1))
    log_V_anti[:, 0] = np.log(V0)
    for i in range(n_steps):
        log_V_anti[:, i+1] = log_V_anti[:, i] + log_inc_anti[:, i]
    V_bar2 = np.mean(np.exp(log_V_anti), axis=1)
    P2 = bs_call_vec(S0, K, r, T, np.sqrt(V_bar2))
    
    return np.mean((P1 + P2) / 2)

def mc_rho_nonzero(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims):
    """MC ρ≠0 VECTORISÉ : simule S et V conjointement, antithétiques sur u."""
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
        V1 = np.maximum(V1 * np.exp((mu - xi**2/2) * dt + xi * sqrt_dt * (rho * u_i + sqrt_1_rho2 * v_i)), 1e-12)
        
        S2 = S2 * np.exp((r - V2/2) * dt + np.sqrt(np.maximum(V2, 0)) * sqrt_dt * (-u_i))
        V2 = np.maximum(V2 * np.exp((mu - xi**2/2) * dt + xi * sqrt_dt * (rho * (-u_i) + sqrt_1_rho2 * v_i)), 1e-12)
    
    p1 = np.exp(-r * T) * np.maximum(S1 - K, 0)
    p2 = np.exp(-r * T) * np.maximum(S2 - K, 0)
    return np.mean((p1 + p2) / 2)

# Paramètres de base
S0 = 1.0
r = 0.0
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
n_steps = 90
n_sims = 10000

# Grille de moneyness
moneyness = np.arange(0.88, 1.13, 0.03)

def compute_smile(rho, xi, T, label=""):
    t0 = time.time()
    sigma_imps = []
    for sx in moneyness:
        K = S0 / sx
        if abs(rho) < 1e-10:
            price = mc_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims)
        else:
            price = mc_rho_nonzero(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims)
        sig = implied_vol_bisection(S0, K, r, T, price)
        sigma_imps.append(sig * 100 if not np.isnan(sig) else np.nan)
    elapsed = time.time() - t0
    if label:
        print(f"  {label} terminé ({elapsed:.1f}s)")
    return sigma_imps

# =====================================================
# (a) Smile de référence (ρ=0, ξ=1, T=180j)
# =====================================================
print("(a) Smile de référence...")
T_ref = 180/365
smile_ref = compute_smile(0.0, 1.0, T_ref, "ρ=0, ξ=1")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
valid = [not np.isnan(s) for s in smile_ref]
ax.plot(moneyness[valid], np.array(smile_ref)[valid], 'bo-', linewidth=2, markersize=5)
ax.axhline(sigma0*100, color='red', linestyle='--', alpha=0.7, label='σ₀ = 15%')
ax.set_xlabel('S / X', fontsize=12)
ax.set_ylabel('σ_imp (%)', fontsize=12)
ax.set_title('(a) Smile de référence (ρ=0, ξ=1, T=180j)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# =====================================================
# (b) Effet de ξ
# =====================================================
print("\n(b) Effet de ξ...")
ax = axes[0, 1]
for xi_val, color in [(0.5, 'green'), (1.0, 'blue'), (2.0, 'red')]:
    smile = compute_smile(0.0, xi_val, T_ref, f"ξ={xi_val}")
    valid = [not np.isnan(s) for s in smile]
    ax.plot(moneyness[valid], np.array(smile)[valid], 'o-', color=color,
            linewidth=2, markersize=4, label=f'ξ = {xi_val}')
ax.axhline(sigma0*100, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('S / X', fontsize=12)
ax.set_ylabel('σ_imp (%)', fontsize=12)
ax.set_title('(b) Effet de ξ (ρ=0, T=180j)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# =====================================================
# (c) Effet de ρ
# =====================================================
print("\n(c) Effet de ρ...")
ax = axes[1, 0]
colors_rho = {-1.0: 'darkblue', -0.5: 'royalblue', 0.0: 'green', 0.5: 'orange', 1.0: 'red'}
for rho_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    smile = compute_smile(rho_val, 1.0, T_ref, f"ρ={rho_val}")
    valid = [not np.isnan(s) for s in smile]
    ax.plot(moneyness[valid], np.array(smile)[valid], 'o-', color=colors_rho[rho_val],
            linewidth=2, markersize=4, label=f'ρ = {rho_val}')
ax.axhline(sigma0*100, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('S / X', fontsize=12)
ax.set_ylabel('σ_imp (%)', fontsize=12)
ax.set_title('(c) Effet de ρ (ξ=1, T=180j)', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# =====================================================
# (d) Effet de T
# =====================================================
print("\n(d) Effet de T...")
ax = axes[1, 1]
for T_days, color in [(90, 'blue'), (180, 'green'), (270, 'red')]:
    T_val = T_days / 365
    smile = compute_smile(0.0, 1.0, T_val, f"T={T_days}j")
    valid = [not np.isnan(s) for s in smile]
    ax.plot(moneyness[valid], np.array(smile)[valid], 'o-', color=color,
            linewidth=2, markersize=4, label=f'T = {T_days} jours')
ax.axhline(sigma0*100, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('S / X', fontsize=12)
ax.set_ylabel('σ_imp (%)', fontsize=12)
ax.set_title('(d) Effet de T (ρ=0, ξ=1)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Smiles de volatilité implicite — Modèle de Hull & White', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('exp8_smiles.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure sauvegardée: exp8_smiles.png")