import pytest
from src.decorators import dgp, method
from jax import numpy as jnp
import jax
import inspect
from src.constants import VALID_KEY_NAMES
import textwrap


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


@dgp(output="X", label="normal")
def norm_data(prng_key, n=100, p=10):
    """
    test docstring
    """
    X = jax.random.normal(prng_key, shape=(n, p))
    return X


def test_dgp(key):
    samples = norm_data(key, n=10, p=5)
    assert samples.shape == (10, 5)
    assert norm_data.label == "normal"


@pytest.mark.parametrize("fn", [norm_data, ridge])
def test_docstring_preservation(fn):
    tmp = inspect.getdoc(fn)
    assert tmp == "test docstring"


def test_vmappable(key):
    key_array = jax.random.split(key, 5)
    assert jax.vmap(norm_data, in_axes=(0,))(key_array).shape == (5, 100, 10)


def test_jitable(key):
    # you can't change the n or p, since shapes are really important for jit compilation I suppose. But this
    # is a JAX thing, not our package, and you can at least JIT any DGP function
    simple_norm = jax.jit(norm_data, static_argnames=("n", "p"))
    assert simple_norm(key, n=10, p=3) is not None


def test_wrong_key_name():
    with pytest.raises(ValueError):

        @dgp(output="test")
        def bad_fn(wrong_key_name, n=10):
            return jnp.zeros((n, 1))


@pytest.mark.parametrize("deco_class", [dgp, method])
@pytest.mark.parametrize("valid_key_name", VALID_KEY_NAMES)
def test_right_key_wrong_place(deco_class, valid_key_name):
    with pytest.raises(ValueError):
        my_fn = f"""
        @{deco_class.__name__}(output="test")
        def bad_fn(n, {valid_key_name}):
            return jnp.zeros((n, 1))
        """
        exec(textwrap.dedent(my_fn))
