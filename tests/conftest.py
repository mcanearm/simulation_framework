import jax
import pytest

from src.decorators import dgp


@pytest.fixture
def key():
    return jax.random.key(0)


@dgp(output="X", label="normal")
def norm_data(prng_key, n=100, p=10):
    """
    test docstring
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    return X
