import numpy as np
from scipy.sparse import diags, identity, csc_matrix
from scipy.sparse.linalg import spsolve
from closed_form_bs import bs_analytic_call

def crank_nicolson_bs_call(S0, K, r, sigma, T, x_min=None, x_max=None, M=400, N=400):
    """
    Crank-Nicolson solver on log-price x = ln S for European call.
    Returns interpolated price at S0

    Params:
    S0: float - current spot
    K: float - strike
    r: float - risk-free rate
    simga:float - volatility
    T: float - maturity (years)
    M: int - number of asset grid intervals (i.e. grid poitns = M + 1)
    N: int - number of time stpes (for tau in [0, T])
    
    Returns: 
    price_interp: float - interpolated price at S0
    """
    lnS0 = np.log(S0)
    if x_min is None:
        x_min = lnS0 - 6.0
    if x_max is None:
        x_max = lnS0 + 6.0
    
    dx = (x_max - x_min) / M
    x = np.linspace(x_min, x_max, M+1)
    tau = np.linspace(0.0, T, N+1)
    dt = tau[1] - tau[0]

    S_grid = np.exp(x)
    u = np.maximum(S_grid - K, 0.0)

    m_interior = M - 1

    #build L 
    A = (sigma**2 / 2.0) / dx**2 - (r - 0.5 * sigma**2) / (2 * dx)    
    C = (sigma**2 / 2.0) / dx**2 + (r - 0.5 * sigma**2) / (2 * dx)
    B = -2 * (sigma**2 / 2.0) / dx**2 - r

    A_vec = A * np.ones(m_interior-1)
    C_vec = C * np.ones(m_interior-1)  
    B_vec = B * np.ones(m_interior)    
    L = diags([A_vec, B_vec, C_vec], offsets=[-1, 0, 1], format='csc')

    # CN matrices
    I = identity(m_interior, format='csc')
    M1 = (I - 0.5 * dt * L).tocsc()
    M2 = (I + 0.5 * dt * L).tocsc()

    # time-stepping
    u_interior = u[1:-1].copy()  
    for n in range(N):
        tau_n = tau[n]
        tau_np1 = tau[n+1]

        # boundary values (Dirichlet)
        u_left = 0.0
        S_right = np.exp(x[-1])
        u_right_n   = S_right - K*np.exp(-r * tau_n)
        u_right_np1 = S_right - K*np.exp(-r * tau_np1)

        RHS = M2.dot(u_interior)

        RHS[0]  += A * u_left
        RHS[-1] += C * u_right_n

        RHS[-1] += 0.5 * dt * C * u_right_np1

        u_interior = spsolve(M1, RHS)

        u[0] = u_left
        u[1:-1] = u_interior
        u[-1] = u_right_np1

    price = np.interp(np.log(S0), x, u)
    return price, {'x': x, 'tau': tau, 'u': u}

if __name__ == "__main__":
    S0 = 100.0; K = 100.0; r = 0.01; sigma = 0.2; T = 1.0
    price_cn, data = crank_nicolson_bs_call(S0, K, r, sigma, T, M = 400, N = 400)
    price_bs = bs_analytic_call(S0, K, r, sigma, T)
    print("CN price:", price_cn)
    print("BS analytic:", price_bs)
    print("Error:", abs(price_cn - price_bs))