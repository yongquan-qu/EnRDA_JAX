# Reference:  Sagar K. Tamang (February 2021) https://github.com/tamangsk/EnRDA/tree/main
# Adaped by: Yongquan Qu

import jax.numpy as jnp


def lorenz63_rhs(x, sigma, rho, beta):
    """
    Computes the right-hand side of the Lorenz-63 system.
    
    Parameters:
        x (jnp.ndarray): State vector [x, y, z]
        sigma (float): Parameter sigma
        rho (float): Parameter rho
        beta (float): Parameter beta
    
    Returns:
        jnp.ndarray: Time derivatives [dx/dt, dy/dt, dz/dt]
    """
    dx = -sigma * (x[0] - x[1])
    dy = rho * x[0] - x[1] - x[0] * x[2]
    dz = x[0] * x[1] - beta * x[2]
    
    return jnp.array([dx, dy, dz])

def lorenz63(xin, dt, parm):
    """
    Integrates the Lorenz-63 system using the RK4 method.
    
    Parameters:
        xin (jnp.ndarray): Initial state vector [x, y, z]
        dt (float): Time step
        parm (jnp.ndarray): Parameters [sigma, rho, beta]
    
    Returns:
        jnp.ndarray: Updated state vector after dt
    """
    sigma, rho, beta = parm

    # Runge-Kutta weights
    w1, w2, w3, w4 = 1/6, 1/3, 1/3, 1/6
    
    # Compute RK4 steps
    k1 = dt * lorenz63_rhs(xin, sigma, rho, beta)
    k2 = dt * lorenz63_rhs(xin + 0.5 * k1, sigma, rho, beta)
    k3 = dt * lorenz63_rhs(xin + 0.5 * k2, sigma, rho, beta)
    k4 = dt * lorenz63_rhs(xin + k3, sigma, rho, beta)
    
    # RK4 update
    xout = xin + w1 * k1 + w2 * k2 + w3 * k3 + w4 * k4
    
    return xout