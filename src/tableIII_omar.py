"""
Reproduction de la Table III du papier
Implied Volatility Calculated by Black-Scholes Formula from the
Option Prices Given in Table II; Actual Expected Mean Volatility
15%; Option Parameters: sigma_0 = 15%, r = 0, xi = 1, and mu = 0

Differences from tableIII.py:
- rho=0: uses mc_rho0 (V-only simulation + analytical BS price), N=100000
- rho!=0: uses mc_joint (joint S,V simulation), N=50000
"""

import numpy as np
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

    payoffs = (P1 + P2) / 2
    return np.mean(payoffs), np.std(payoffs) / np.sqrt(n_sims)

def mc_joint(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims):
    """MC vectorise, simulation conjointe de S et V, antithetiques sur u."""
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

# Parametres
S0 = 1.0
r = 0.0
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 1.0
n_steps = 90
n_sims_rho0 = 100000
n_sims_joint = 50000

SX_values = [0.90, 0.95, 1.00, 1.05, 1.10]
rho_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
T_configs = [(90, 90/365), (180, 180/365), (270, 270/365)]

lines = []
lines.append("Table III")
lines.append("Implied Volatility Calculated by Black-Scholes Formula from the")
lines.append("Option Prices Given in Table II; Actual Expected Mean Volatility")
lines.append(f"15%; Option Parameters: sigma_0 = {sigma0*100:.0f}%, r = {r}, xi = {xi:.0f}, and mu = {mu:.0f}")
lines.append("")

for l in lines:
    print(l)

for T_days, T in T_configs:
    sx_header = "".join(f"{sx:>10}" for sx in SX_values)
    block = []
    block.append(f"T = {T_days} Days")
    block.append(f"{'rho':>6}{sx_header}")
    block.append("-" * (6 + 10 * len(SX_values)))

    for b in block:
        print(b)
    lines.extend(block)

    for rho in rho_values:
        t0 = time.time()
        vol_line = f"{rho:6.1f}"
        se_line = f"{'':>6}"

        for sx in SX_values:
            K = S0 / sx

            if abs(rho) < 1e-10:
                phw, se_price = mc_rho0(S0, K, r, T, V0, xi, n_steps, n_sims_rho0)
            else:
                phw, se_price = mc_joint(S0, K, r, T, V0, mu, xi, rho, n_steps, n_sims_joint)

            sig_imp = implied_vol_bisection(S0, K, r, T, phw)

            if not np.isnan(sig_imp):
                # SE sur la vol implicite via delta method : se_vol = se_price / vega
                d1 = (np.log(S0/K) + (r + sig_imp**2/2)*T) / (sig_imp * np.sqrt(T))
                vega = S0 * np.sqrt(T) * norm.pdf(d1)
                if vega > 1e-10:
                    se_vol = se_price / vega * 100
                else:
                    se_vol = np.nan
                vol_line += f"{sig_imp*100:10.2f}"
                se_line += f"{'(' + f'{se_vol:.2f}' + ')':>10}" if not np.isnan(se_vol) else f"{'(??)':>10}"
            else:
                vol_line += f"{'******':>10}"
                se_line += f"{'******':>10}"

        elapsed = time.time() - t0
        print(vol_line)
        print(se_line)
        lines.append(vol_line)
        lines.append(se_line)
        print(f"  [rho={rho:+.1f}, {elapsed:.1f}s]")

    print()
    lines.append("")

filepath = os.path.join(output_dir, 'table_III.txt')
with open(filepath, 'w') as f:
    f.write('\n'.join(lines))

print(f"Sauvegarde: {filepath}")
