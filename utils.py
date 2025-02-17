# Reference:  Sagar K. Tamang (February 2021) https://github.com/tamangsk/EnRDA/tree/main
# Adaped by: Yongquan Qu


import jax.numpy as jnp


def entrop_OMT(x, y, p, q, gamma, niter):
    """
    Approximates the optimal transport plan using entropic regularization.
    
    Parameters:
        x (jnp.ndarray): Source points (d, N1)
        y (jnp.ndarray): Target points (d, N2)
        p (jnp.ndarray): Source probability distribution (N1,)
        q (jnp.ndarray): Target probability distribution (N2,)
        gamma (float): Regularization parameter
        niter (int): Number of Sinkhorn iterations
    
    Returns:
        U (jnp.ndarray): Optimal transport matrix
    """
    x = x.T  # Shape (d, N1) -> (N1, d)
    y = y.T  # Shape (d, N2) -> (N2, d)
    
    N1, d1 = x.shape
    N2, d2 = y.shape

    # Compute squared Euclidean cost matrix
    x2 = jnp.sum(x**2, axis=1, keepdims=True)  # Shape (N1, 1)
    y2 = jnp.sum(y**2, axis=1, keepdims=True)  # Shape (N2, 1)
    C = x2 + y2.T - 2 * jnp.dot(x, y.T)  # Shape (N1, N2)
    
    # Compute kernel matrix K
    K = jnp.exp(-C / gamma)  # Shape (N1, N2)
    
    # Sinkhorn iteration
    b = jnp.ones((N2,))  
    for _ in range(niter):
        a = p / (K @ b)
        b = q / (K.T @ a)
    
    U = jnp.diag(a) @ K @ jnp.diag(b)
    
    return U



def covariance(X, Y):
    """
    Computes the covariance matrix between X and Y.
    
    Parameters:
        X (jnp.ndarray): Data matrix of shape (d, N)
        Y (jnp.ndarray): Data matrix of shape (d, N)
    
    Returns:
        C (jnp.ndarray): Covariance matrix of shape (d, d)
    """
    N = X.shape[1]
    
    # Compute means along the second axis
    X_bar = jnp.mean(X, axis=1, keepdims=True)  # Shape (d, 1)
    Y_bar = jnp.mean(Y, axis=1, keepdims=True)  # Shape (d, 1)

    # Compute covariance
    C = (X - X_bar) @ (Y - Y_bar).T / (N - 1)

    return C