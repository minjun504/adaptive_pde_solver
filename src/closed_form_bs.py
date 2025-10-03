import numpy as np
from scipy.stats import norm 

def bs_analytic_call(S, K, r, sigma, T):
    """
    Black-Scholes closed-form Eueropean call price
    To calculate the baseline comparison price of a call option
    """
    if T == 0:
        return np.maximum(S-K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

if __name__ == "__main__":
    None


