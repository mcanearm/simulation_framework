from src.decorators import dgp, method
from jax import numpy as jnp
from src.runners import run_methods
import jax
import pytest


@method(output="beta", label="Ridge")
def ridge(X, y, alpha=1.0):
    """
    Fit a ridge regression model.
    """
    p = X.shape[1]
    beta_hat = jnp.linalg.inv(X.T @ X + alpha * jnp.eye(p)) @ X.T @ y
    return beta_hat


@dgp(output=["X", "y"], label="linear_data")
def linear_data(prng_key, n=100, p=10):
    """
    Generate linear regression data.
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    true_beta = jnp.arange(1, p + 1)
    noise = jax.random.normal(prng_key, shape=(n,)) * 0.5
    y = X @ true_beta + noise
    return X, y


@pytest.mark.skip()
def test_runner():
    key = jax.random.PRNGKey(0)
    n_sims = 5

    simulation_output = run_methods(key, dgp=linear_data, method=ridge, n_sims=n_sims)
