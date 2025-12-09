import jax
import pytest
from jax import numpy as jnp

from src.decorators import dgp, method


# TODO: all these functions and fixtures are using the wrong interface for
# random keys in JAX. Need to migrate to jax.random.key(), which coinceidentally,
# will align nicely with numpy and should allow numpy RNG support as well.
@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@method(output="beta_hat", label="Ridge")
def ridge(X, y, alpha=0.1):
    """
    test docstring
    """
    p = X.shape[1]
    beta_hat = jnp.linalg.inv(X.T @ X + alpha * jnp.eye(p)) @ X.T @ y
    return beta_hat


@method(output="beta_hat", label="OLS")
def ols(X, y):
    """
    Ordinary Least Squares regression
    """
    beta_hat = jnp.linalg.inv(X.T @ X) @ X.T @ y
    return beta_hat


@dgp(output=["X", "y", "beta"], label="OLS")
def ols_data(prng_key, n=100, p=10, noise=1.0, dist="normal"):
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
    return X, y, true_beta


@dgp(output="X", label="normal")
def norm_data(prng_key, n=100, p=10):
    """
    test docstring
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    return X


@dgp(output=["X", "y", "beta"], label="exponential_data")
def exponential_data(prng_key, n=100, p=10, noise=1.0):
    """
    Generate data with an exponential relationship.
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    true_beta = jnp.arange(1, p + 1)
    noise = jax.random.normal(prng_key, shape=(n,)) * noise
    y = jnp.exp(X @ true_beta / 10) + noise
    return X, y, true_beta
