import pytest
import jax
from jax import numpy as jnp
from src.decorators import method, dgp


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@method(output="beta", label="Ridge")
def ridge(X, y):
    """
    test docstring
    """
    p = X.shape[1]
    alpha = 1.0
    beta_hat = jnp.linalg.inv(X.T @ X + alpha * jnp.eye(p)) @ X.T @ y
    return beta_hat


@method(output="beta", label="OLS")
def ols(X, y):
    """
    test docstring
    """
    beta_hat = jnp.linalg.inv(X.T @ X) @ X.T @ y
    return beta_hat


@dgp(output="X", label="normal")
def norm_data(prng_key, n=100, p=10):
    """
    test docstring
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    return X
