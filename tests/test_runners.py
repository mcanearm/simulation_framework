from src.decorators import dgp, method
from jax import numpy as jnp
from src.runners import run_methods
import jax
import pytest
from src.utils import simulation_grid


@method(output="beta", label="Ridge")
def ridge(X, y, alpha=1.0):
    """
    Fit a ridge regression model.
    """
    p = X.shape[1]
    beta_hat = jnp.linalg.inv(X.T @ X + alpha * jnp.eye(p)) @ X.T @ y
    return beta_hat


@dgp(output=["X", "y"], label="linear_data")
def linear_data(prng_key, n=100, p=10, noise=1.0, dist="normal"):
    """
    Generate linear regression data.
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    true_beta = jnp.arange(1, p + 1)
    if dist == "normal":
        noise = jax.random.normal(prng_key, shape=(n,)) * noise
    elif dist == "t":
        noise = jax.random.t(prng_key, df=3, shape=(n,)) * noise
    noise = jax.random.normal(prng_key, shape=(n,)) * 0.5
    y = X @ true_beta + noise
    return X, y


def exponential_data(prng_key, n=100, p=10, noise=1.0):
    """
    Generate data with an exponential relationship.
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    true_beta = jnp.arange(1, p + 1)
    noise = jax.random.normal(prng_key, shape=(n,)) * noise
    y = jnp.exp(X @ true_beta / 10) + noise
    return X, y


# scenarios = simulation_grid(
#     dgps=[(linear_data, {"n": [50, 100], "p": [1, 5, 20], "dist": ["normal", "t"]})],
#     methods=[(ridge, {"alpha": [0.1, 1.0, 10.0]})(ols, {})],
# )
