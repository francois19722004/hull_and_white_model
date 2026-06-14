"""
Expérience 2 — Validation ξ = 0
Vérifie que quand ξ=0 (vol constante), le MC retrouve le prix B-S exact
et que le smile est plat à σ₀.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_call(S, K, r, T, sigma):
    if sigma <= 1e-12 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol_bisection(S, K, r, T, price, tol=1e-8, max_iter=200):
    """Inversion de B-S par dichotomie."""
    lower_bound = max(S - K * np.exp(-r * T), 0.0)
    if price <= lower_bound + 1e-12 or price >= S - 1e-12:
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

def mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims, antithetic=True):
    """Monte Carlo Hull-White, cas rho=0 : simule V seul."""
    dt = T / n_steps
    prices = np.zeros(n_sims)
    
    for k in range(n_sims):
        z = np.random.randn(n_steps)
        
        # Trajectoire directe
        V = np.zeros(n_steps + 1)
        V[0] = V0
        for i in range(n_steps):
            V[i+1] = V[i] * np.exp((mu - xi**2 / 2) * dt + xi * np.sqrt(dt) * z[i])
        V_bar1 = np.mean(V)
        P1 = bs_call(S0, K, r, T, np.sqrt(max(V_bar1, 1e-12)))
        
        if antithetic:
            # Trajectoire antithétique
            V_anti = np.zeros(n_steps + 1)
            V_anti[0] = V0
            for i in range(n_steps):
                V_anti[i+1] = V_anti[i] * np.exp((mu - xi**2 / 2) * dt + xi * np.sqrt(dt) * (-z[i]))
            V_bar2 = np.mean(V_anti)
            P2 = bs_call(S0, K, r, T, np.sqrt(max(V_bar2, 1e-12)))
            prices[k] = (P1 + P2) / 2
        else:
            prices[k] = P1
    
    return np.mean(prices), np.std(prices) / np.sqrt(n_sims)

# Paramètres
S0 = 100.0
r = 0.05
T = 0.5
sigma0 = 0.15
V0 = sigma0**2
mu = 0.0
xi = 0.0  # <-- VOL-OF-VOL NULLE
n_steps = 50
n_sims = 5000

# 4 strikes seulement : OTM, légèrement OTM, ATM, ITM
moneyness = np.array([0.85, 0.95, 1.00, 1.10])
strikes = moneyness * S0

print("=" * 70)
print("VALIDATION ξ = 0 : le MC doit retrouver le prix B-S exact")
print("=" * 70)
print(f"{'K/S0':>6} {'Prix BS':>10} {'Prix MC':>10} {'Std Err':>10} {'σ_imp':>10} {'Écart':>10}")
print("-" * 70)

sigma_imps = []
for K in strikes:
    price_bs = bs_call(S0, K, r, T, sigma0)
    price_mc, std_err = mc_hull_white_rho0(S0, K, r, T, V0, mu, xi, n_steps, n_sims)
    sig_imp = implied_vol_bisection(S0, K, r, T, price_mc)
    sigma_imps.append(sig_imp)
    ecart = (price_mc - price_bs) / price_bs * 100 if price_bs > 1e-8 else 0
    print(f"{K/S0:6.2f} {price_bs:10.4f} {price_mc:10.4f} {std_err:10.6f} "
          f"{sig_imp*100 if not np.isnan(sig_imp) else 0:10.2f}% {ecart:10.4f}%")

# Tracer le smile (devrait être plat)
fig, ax = plt.subplots(figsize=(10, 6))
valid = [not np.isnan(s) for s in sigma_imps]
m_valid = moneyness[valid]
s_valid = np.array(sigma_imps)[valid] * 100
ax.plot(m_valid, s_valid, 'bo-', markersize=8, linewidth=2, label='σ_imp (MC, ξ=0)')
ax.axhline(sigma0 * 100, color='red', linestyle='--', linewidth=2, label=f'σ₀ = {sigma0*100:.0f}%')
ax.set_xlabel('K / S₀ (moneyness)', fontsize=13)
ax.set_ylabel('Volatilité implicite (%)', fontsize=13)
ax.set_title('Validation ξ = 0 : le smile doit être plat à σ₀ = 15%', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(10, 20)
plt.tight_layout()
plt.savefig('exp2_validation_xi0.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure sauvegardée: exp2_validation_xi0.png")