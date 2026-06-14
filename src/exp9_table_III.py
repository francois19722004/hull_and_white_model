"""
Expérience 9 — Reproduction de la Table III du papier
Volatilités implicites par S/X, ρ et T.
Paramètres : σ₀ = 15%, ξ = 1, μ = 0, r = 0.
"""

import numpy as np
from scipy.stats import norm
import time

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

def mc_rho_nonzero(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims):
    """MC ρ≠0 VECTORISÉ."""
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
    payoffs = (p1 + p2) / 2
    return np.mean(payoffs), np.std(payoffs) / np.sqrt(n_sims)

def mc_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims):
    """MC ρ=0 VECTORISÉ."""
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
    
    prices = (P1 + P2) / 2
    return np.mean(prices), np.std(prices) / np.sqrt(n_sims)

# Paramètres
S0 = 1.0
r = 0.0
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 1.0
n_steps = 90
n_sims = 8000

SX_values = [0.90, 0.95, 1.00, 1.05, 1.10]
rho_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
T_values = [(90, 90/365), (180, 180/365), (270, 270/365)]

print("=" * 90)
print("TABLE III — Volatilités implicites calculées par inversion de Black-Scholes")
print(f"Volatilité attendue : σ₀ = {sigma0*100:.0f}%")
print(f"Paramètres : ξ = {xi}, μ = {mu}, r = {r}")
print("=" * 90)

t_global = time.time()

for T_days, T in T_values:
    print(f"\n{'='*80}")
    print(f"T = {T_days} jours")
    print(f"{'ρ':>6}", end="")
    for sx in SX_values:
        print(f"{'S/X='+str(sx):>14}", end="")
    print()
    print("-" * 80)
    
    for rho in rho_values:
        t0 = time.time()
        print(f"{rho:6.1f}", end="")
        for sx in SX_values:
            K = S0 / sx
            
            if abs(rho) < 1e-10:
                price, se = mc_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims)
            else:
                price, se = mc_rho_nonzero(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims)
            
            sig_imp = implied_vol_bisection(S0, K, r, T, price)
            if not np.isnan(sig_imp):
                vega = S0 * np.sqrt(T) * norm.pdf(
                    (np.log(S0/K) + (r + sig_imp**2/2)*T) / (sig_imp * np.sqrt(T)))
                if vega > 1e-10:
                    se_imp = se / vega
                else:
                    se_imp = np.nan
                print(f"  {sig_imp*100:6.2f}({se_imp*100:.2f})" if not np.isnan(se_imp) else f"  {sig_imp*100:6.2f}(  ??)", end="")
            else:
                print(f"       ******", end="")
        elapsed = time.time() - t0
        print(f"  [{elapsed:.1f}s]")

print(f"\nTemps total: {time.time() - t_global:.0f}s")
print("\nComparer avec le papier (Table III, p.297):")
print("  Pour ρ=0, T=90j, ATM: papier donne 14.86%")
print("  L'effet maturité: pour ρ=0, σ_imp ATM décroît quand T augmente")